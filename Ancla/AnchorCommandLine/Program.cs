using Database;
using Giraffe.ViewEngine;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Data;
using Proteomics.PSM;
using Readers;

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
    public Dictionary<string, double> LibraryRetentionTimes = new Dictionary<string, double>();
    public List<FileLogger> FileLoggers = new List<FileLogger>();
    public Calibrator(string filePath)
    {
        // read the psmtsv file
        var psmtsv = new PsmFromTsvFile(filePath);
        psmtsv.LoadResults();

        // make the calibration logger
        FileLoggers.Add(new FileLogger(psmtsv));
    }

    public void Calibrate()
    {
        foreach (var fileLogger in FileLoggers)
        {
            fileLogger.Calibrate();
        }
    }

    //this method should return a list of dictionaries where the key is the full sequence and the value is the retention time
    public Dictionary<string, Dictionary<string, double>> GetCalibrations()
    {
        Dictionary<string, Dictionary<string, double>> calibrations = new Dictionary<string, Dictionary<string, double>>();

        foreach (var rawFile in FileLoggers[0].FollowingRawFiles)
        {
            Dictionary<string, double> rawFileSequences = new Dictionary<string, double>();

            foreach (var fullSequence in rawFile.FullSequenceWithScanRetentionTime)
            {
                rawFileSequences.Add(fullSequence.Key, fullSequence.Value);
            }
            //sort by value
            rawFileSequences = rawFileSequences.OrderBy(p => p.Value).ToDictionary(p => p.Key, p => p.Value);

            calibrations.Add(rawFile.RawFileName, rawFileSequences);
        }

        return calibrations;
    }
}

public class FileLogger
{
    public string FilePath { get; set; }
    public PsmFromTsvFile File { get; set; }
    public RawFileLogger LeadingRawFile { get; set; }
    public List<RawFileLogger> FollowingRawFiles = new();
    public Dictionary<string, RawFileLogger> RawFiles = new Dictionary<string, RawFileLogger>();
    public Dictionary<string, List<(string, double?)>> FullSequencesPresentInFile = new Dictionary<string, List<(string, double?)>>();
    public Dictionary<string, List<(string, double)>> FileWiseCalibrations = new Dictionary<string, List<(string, double)>>();
    public FileLogger(PsmFromTsvFile file)
    {
        File = file;
        FilePath = file.FilePath;

        var rawFiles = file.Results.GroupBy(p => p.FileNameWithoutExtension);

        // sorts the psms by raw file name, where the RawFiles dictionary key is the raw file name and the value is a RawFileLogger object 
        foreach (var rawFileName in rawFiles)
        {
            RawFiles.Add(rawFileName.Key, new RawFileLogger(rawFileName.Key, rawFileName.Select(p => p)));
        }

        // Get a list of all the full sequences present in the file
        List<string> fullSequences = file.Results.Select(p => p.FullSequence)
                                                 .Distinct()
                                                 .ToList();

        // Retention Times will be initialized as null
        FullSequencesPresentInFile =
            fullSequences.ToDictionary(p => p, p => new List<(string, double?)>());

        // pick the leading raw file and set the follower raw files
        LeadingRawFile = RawFiles.Values.OrderBy(r => r.Psms.Count()).First();
        FollowingRawFiles = RawFiles.Values.Where(r => r != LeadingRawFile).ToList();
    }

    public void Calibrate()
    {
        foreach (var follower in FollowingRawFiles)
        {
            PairwiseCalibration(follower);
        }
    }

    private void PairwiseCalibration(RawFileLogger followingRawFile)
    {
        // Get overlapping peptides between the leading and following raw file
        var overlappingFullSequences = LeadingRawFile.FullSequenceWithScanRetentionTime.Keys
            .Intersect(followingRawFile.FullSequenceWithScanRetentionTime.Keys);

        Dictionary<string, (double, double)> overlappingPsms = new();

        foreach (var sequence in overlappingFullSequences)
        {
            overlappingPsms.Add(sequence, (LeadingRawFile.FullSequenceWithScanRetentionTime[sequence],
                followingRawFile.FullSequenceWithScanRetentionTime[sequence]));
        }

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
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

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

            if(!FileWiseCalibrations.ContainsKey(fullSequence.Key))
            {
                FileWiseCalibrations.Add(fullSequence.Key, new List<(string, double)>());
            }

            // update the retention time of the full sequence in the following raw file
            FileWiseCalibrations[fullSequence.Key].Add((followingRawFile.RawFileName + "_Calibrated",
                prediction.TransformedRetentionTime));
        }

        // for each full sequence in the leading raw file, insert those that are not present in the following raw file

        foreach (var fullSequence in LeadingRawFile.FullSequenceWithScanRetentionTime)
        {
            if (!FileWiseCalibrations.ContainsKey(fullSequence.Key))
            {
                FileWiseCalibrations.Add(fullSequence.Key, new List<(string, double)>());
            }

            FileWiseCalibrations[fullSequence.Key].Add((LeadingRawFile.RawFileName + "_OG", fullSequence.Value));
        }
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

public class RawFileLogger
{
    public string RawFileName { get; set; }
    public IEnumerable<PsmFromTsv> Psms { get; set; }
    public Dictionary<string, double> FullSequenceWithScanRetentionTime = new Dictionary<string, double>();
    public RawFileLogger(string rawFileName, IEnumerable<PsmFromTsv> psms)
    {
        RawFileName = rawFileName;
        // TODO Filter the psms
        Psms = psms.Where(p => p.QValue <= 0.01 &
                               p.PEP <= 0.5 &
                               p.AmbiguityLevel == "1" &
                               p.DecoyContamTarget == "T").ToList();

        // Get the median retention time for each full sequence that are repeated in the raw file
        var fullSequences = Psms.GroupBy(p => p.FullSequence);
        foreach (var fullSequence in fullSequences)
        {
            FullSequenceWithScanRetentionTime.Add(fullSequence.Key, fullSequence.Select(p => p.RetentionTime).Median());
        }

        //TODO calibrate raw file 
    }


}

public class DatabaseConnection
{
    public string ConnectionString { get; set; }
}