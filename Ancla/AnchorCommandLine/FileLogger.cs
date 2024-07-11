using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Readers;

namespace AnchorCommandLine;

public class FileLogger
{
    public string FilePath { get; set; }
    public PsmFromTsvFile PsmFile { get; set; }
    public RawFileLogger LeadingRawFile { get; set; }
    public List<RawFileLogger> FollowingRawFiles = new();
    public Dictionary<string, RawFileLogger> RawFiles = new Dictionary<string, RawFileLogger>();
    public Dictionary<string, List<(string, double?)>> FullSequencesPresentInFile = 
        new Dictionary<string, List<(string, double?)>>();
    public Dictionary<string, List<(string fileName, double retentionTime)>> FileWiseCalibrations = 
        new Dictionary<string, List<(string, double)>>();
    public FileLogger(PsmFromTsvFile psmFile)
    {
        PsmFile = psmFile;
        FilePath = psmFile.FilePath;

        var rawFiles = psmFile.Results.GroupBy(p => p.FileNameWithoutExtension);

        // sorts the psms by raw psmFile name, where the RawFiles dictionary key is the raw psmFile name and the value is a RawFileLogger object 
        foreach (var rawFileName in rawFiles)
        {
            RawFiles.Add(rawFileName.Key, new RawFileLogger(rawFileName.Key, rawFileName.Select(p => p)));
        }

        // Get a list of all the full sequences present in the psmFile
        FullSequencesPresentInFile = psmFile.Results.Select(p => p.FullSequence)
            .Distinct()
            .ToDictionary(p => p, p => new List<(string, double?)>());

        // pick the leading raw psmFile and set the follower raw files
        LeadingRawFile = RawFiles.Values.OrderBy(r => r.Psms.Count()).First();
        FollowingRawFiles = RawFiles.Values.Where(r => r != LeadingRawFile).ToList();
    }

    public void Calibrate()
    {
        SaveFullSequencesPresentFileWiseAsTSV();
        RemoveAndRecalibrateAllFiles();
        WriteOutput();

    }

    private void DeleteFileValues(string fileName)
    {
        Dictionary<string, List<(string, double)>> swapDictionary = new Dictionary<string, List<(string, double)>>();

        foreach (var pair in FileWiseCalibrations)
        {
            List<(string, double)> t = pair.Value;

            if (t.Select(s => s.Item1).ToList().Contains(fileName))
            {
                t.RemoveAll(v => v.Item1 == fileName);
            }
            swapDictionary.Add(pair.Key, t);
        }
    }

    private void RemoveAndRecalibrateAllFiles()
    {
        for (int i = 0; i < 10; i++)
        {
            foreach (var filename in RawFiles)
            {
                DeleteFileValues(filename.Key);
                PairwiseCalibration(filename.Value);
            }
        }
    }
    private void SaveFullSequencesPresentFileWiseAsTSV()
    {
        var grouped = PsmFile.GroupBy(x => x.FullSequence)
            .ToDictionary(p => p.Key, p => p
                .DistinctBy(x => x.FileNameWithoutExtension).Select(x => (x.FileNameWithoutExtension, x.RetentionTime.Value)).ToList());

        List<string> myOutput = new List<string>();

        foreach (var pair in grouped)

        {
            List<double> times = pair.Value.Select(s => s.Value).ToList();
            string s = pair.Key + "\t" + times.Median() + "\t" + string.Join("\t", times);
            myOutput.Add(s);
        }
        File.WriteAllLines(@"D:\UnCalibratedFiles_OLS_loops10_filtredBy2.tsv", myOutput);

        FileWiseCalibrations = grouped;
    }

    private void PairwiseCalibration(RawFileLogger followingRawFile)
    {
        // Get overlapping peptides between the leading and following raw psmFile
        var overlappingFullSequences = FileWiseCalibrations.Keys
            .Intersect(followingRawFile.FullSequenceWithScanRetentionTime.Keys);

        var bubba = FileWiseCalibrations
            .Where(v => v.Value.Count > 2)
            .ToDictionary(p => p.Key, p => p);

        Dictionary<string, (double median, double)> overlappingPsms = bubba.Keys
            .Intersect(followingRawFile.FullSequenceWithScanRetentionTime.Keys)
            .ToDictionary(p => p, p => (FileWiseCalibrations[p].Select(x => x.retentionTime).Median(),
                followingRawFile.FullSequenceWithScanRetentionTime[p]));

        // use ml.net to train a linear regression model using the leader and follower retention times as training data
        MLContext mlContext = new MLContext();
        var data = new List<Anchor>();

        foreach (var overlappingPsm in overlappingPsms)
        {
            data.Add(new Anchor
            {
                FullSequence = overlappingPsm.Key,
                LeaderRetentionTime = (float)overlappingPsm.Value.Item1,
                FollowerRetentionTime = (float)overlappingPsm.Value.Item2
            });
        }

        var dataView = mlContext.Data.LoadFromEnumerable<Anchor>(data.ToArray());

        var pipeline = mlContext.Transforms
            .CopyColumns("Label", nameof(Anchor.LeaderRetentionTime))
            .Append(mlContext.Transforms.Concatenate("Features", nameof(Anchor.FollowerRetentionTime)))
            .Append(mlContext.Regression.Trainers.Ols(labelColumnName: "Label", featureColumnName: "Features"));

        var model = pipeline.Fit(dataView);

        // use the model to predict the follower retention times
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Anchor, AnchorPrediction>(model);

        foreach (var fullSequence in followingRawFile.FullSequenceWithScanRetentionTime)
        {
            var prediction = predictionEngine.Predict(new Anchor
            {
                FullSequence = fullSequence.Key,
                FollowerRetentionTime = (float)fullSequence.Value
            });

            if (!FileWiseCalibrations.ContainsKey(fullSequence.Key))
            {
                FileWiseCalibrations.Add(fullSequence.Key, new List<(string, double)>());
            }

            // update the retention time of the full sequence in the following raw psmFile
            FileWiseCalibrations[fullSequence.Key].Add((followingRawFile.RawFileName,
                prediction.TransformedRetentionTime));
        }

        // for each full sequence in the leading raw psmFile, insert those that are not present in the following raw psmFile

        foreach (var fullSequence in LeadingRawFile.FullSequenceWithScanRetentionTime)
        {
            if (!FileWiseCalibrations.ContainsKey(fullSequence.Key))
            {
                FileWiseCalibrations.Add(fullSequence.Key, new List<(string, double)>());
            }

            FileWiseCalibrations[fullSequence.Key].Add((LeadingRawFile.RawFileName, fullSequence.Value));
        }
    }

    private void MergeRawFiles()
    {
        var fileWiseCalibrationSwap = new Dictionary<string, List<(string, double)>>();

        // remove FileWiseCalibration keys that have the same tuple item 1 and item 2
        foreach (var fileWiseCalibration in FileWiseCalibrations)
        {
            var uniqueFileNames = fileWiseCalibration.Value.DistinctBy(x => x.Item1).ToList();

            fileWiseCalibrationSwap.Add(fileWiseCalibration.Key, uniqueFileNames);
        }

        FileWiseCalibrations = fileWiseCalibrationSwap;
    }

    private void WriteOutput()
    {
        List<string> myOutput = new List<string>();

        foreach (var pair in FileWiseCalibrations)

        {
            List<double> times = pair.Value.Select(s => s.retentionTime).ToList();
            string s = pair.Key + "\t" + times.Median() + "\t" + string.Join("\t", times);
            myOutput.Add(s);
        }
        File.WriteAllLines(@"D:\CalibratedFiles_OLS_loops10_filtered2.tsv", myOutput);
    }
}