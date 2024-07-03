using Database;
using Easy.Common.Interfaces;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using mzIdentML110.Generated;
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
    public Calibrator(string filePath)
    { 
        // read the psmtsv file
        var psmtsv = new PsmFromTsvFile(filePath);
        psmtsv.LoadResults();

        // make the calibration logger
        var logger = new FileLogger(psmtsv);
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
            fullSequences.ToDictionary(p => p, p => new List<(string, double?)>() { ("Init", (double?)null) });

        // pick the leading raw file and set the follower raw files
        LeadingRawFile = RawFiles.Values.OrderBy(r => r.Psms.Count()).First();
        FollowingRawFiles = RawFiles.Values.Where(r => r != LeadingRawFile).ToList();
    }

    public void Calibrate()
    {
        foreach (var follower in FollowingRawFiles)
        {

        }
    }

    private void PairwiseCalibration(RawFileLogger followingRawFile)
    {
        // Order the full sequences by retention time
        var orderedLeaderFullSequences = LeadingRawFile.FullSequenceWithScanRetentionTime
            .OrderBy(p => p.Value).ToList();

        var orderedFollowerFullSequences = followingRawFile.FullSequenceWithScanRetentionTime
            .OrderBy(p => p.Value).ToList();

        // Get overlapping peptides between the leading and following raw file
        var overlappingPeptides = orderedLeaderFullSequences.Select(p => p.Key)
            .Intersect(orderedFollowerFullSequences.Select(p => p.Key)).ToList();

        // get these full sequences from the leader and follower raw file as an array of doubles
        var leaderRetentionTimes = overlappingPeptides
            .Select(p => (orderedLeaderFullSequences.First(x => x.Key.Equals(p)))).ToList();

        var followerRetentionTimes = overlappingPeptides
            .Select(p => (orderedFollowerFullSequences.First(x => x.Key.Equals(p)))).ToList();

        // assert that the leader and follower retention times are the same length
        if (leaderRetentionTimes.Count != followerRetentionTimes.Count)
        {
            throw new Exception("Leader and follower anchors retention times are not the same length");
        }

        // use ml.net to train a linear regression model using the leader and follower retention times as training data
        MLContext mlContext = new MLContext();
        var data = new List<Anchor>();

        for (int i = 0; i < leaderRetentionTimes.Count; i++)
        {
            data.Add(new Anchor
            {
                FullSequence = overlappingPeptides[i],
                LeaderRetentionTime = leaderRetentionTimes[i].Value,
                FollowerRetentionTime = followerRetentionTimes[i].Value
            });
        }

        var dataView = mlContext.Data.LoadFromEnumerable<Anchor>(data);

        var model = mlContext.Regression.Trainers.Sdca("LeaderRetentionTime", "FollowerRetentionTime");

        var modelTrained = model.Fit(dataView);

        // use the model to predict the follower retention times
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Anchor, AnchorPrediction>(modelTrained);

        foreach (var fullSequence in followingRawFile.FullSequenceWithScanRetentionTime)
        {
            var prediction = predictionEngine.Predict(new Anchor
            {
                FullSequence = fullSequence.Key,
                LeaderRetentionTime = fullSequence.Value
            });

            // update the retention time of the full sequence in the following raw file
            followingRawFile.FullSequenceWithScanRetentionTime[fullSequence.Key] = prediction.Score;
        }
    }
}

public class Anchor
{
    public string FullSequence { get; set; }
    public double LeaderRetentionTime { get; set; }
    public double FollowerRetentionTime { get; set; }
}

public class AnchorPrediction
{
    public float Score { get; set; }
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