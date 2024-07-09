using Database;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Data;
using Readers;
using ThermoFisher.CommonCore.Data.Business;

namespace AnchorCommandLine;
public class Program
{
    public static void Main(string[] args)
    {
        List<string> paths = new List<string>();

        foreach (var arg in args)
        {
            paths.Add(arg);
        }

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            var psms = PsmService.GetPsms(paths);
        }
    }
}

public class Calibrator
{
    /// <summary>
    /// Holds all the RawFiles loaded into the calibrator
    /// </summary>
    public List<RawFileLogger> RawFileLoggers= new();
    
    public List<string> FilePaths = new();
    
    public List<PsmFromTsvFile> PsmFiles = new();

    /// <summary>
    /// Contains all the Full sequences present in all Files
    /// </summary> //TODO: Change the tuple list into a dictionary
    public Dictionary<string, List<(string fileName, double retentionTime)>> FileWiseCalibrations = new();

    /// <summary>
    /// For use if the file contains multiple raw files.
    /// </summary>
    /// <param name="filePath"></param>
    public Calibrator(string filePath)
    {
        // read the psmtsv psmFile
        var psmtsv = new PsmFromTsvFile(filePath);
        psmtsv.LoadResults();

        // make the calibration logger
        RawFileLoggers.Add(new RawFileLogger(psmtsv));
    }

    public Calibrator(List<string> filesPaths)
    {
        foreach (var filePath in filesPaths)
        {
            // read the psmtsv psmFile
            var psmtsv = new PsmFromTsvFile(filePath);
            psmtsv.LoadResults();

            // make the calibration logger
            RawFileLoggers.Add(new RawFileLogger(psmtsv));
        }
    }

    public void Calibrate()
    {
        foreach (var fileLogger in RawFileLoggers)
        {
            fileLogger._calibrate();
        }
    }

    private void _calibrate()
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
                .DistinctBy(x => x.FileNameWithoutExtension)
                .Select(x => (x.FileNameWithoutExtension, x.RetentionTime.Value))
                .ToList());

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

        var bubba = FileWiseCalibrations.Where(v => v.Value.Count > 2)
            .ToDictionary(p => p.Key, p => p);

        Dictionary<string, (double median, double)> overlappingPsms = bubba.Keys
            .Intersect(followingRawFile.FullSequenceWithScanRetentionTime.Keys)
            .ToDictionary(p => p, p => (FileWiseCalibrations[p]
                    .Select(x => x.retentionTime).Median(), followingRawFile.FullSequenceWithScanRetentionTime[p]));

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

public class Anchor
{
    public string FullSequence { get; set; }
    public float LeaderRetentionTime { get; set; }
    public float FollowerRetentionTime { get; set; }
}

public class AnchorPrediction
{
    [ColumnName("Score")]
    public float TransformedRetentionTime { get; set; }
}