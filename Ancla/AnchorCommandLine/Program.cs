using Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML.Data;
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
    /// <summary>
    /// Holds all the RawFiles loaded into the calibrator
    /// </summary>
    public List<FileLogger> FileLoggers = new List<FileLogger>();
    
    public Calibrator(string filePath)
    {
        // read the psmtsv psmFile
        var psmtsv = new PsmFromTsvFile(filePath);
        psmtsv.LoadResults();

        // make the calibration logger
        FileLoggers.Add(new FileLogger(psmtsv));
    }

    public Calibrator(List<string> filesPaths)
    {
        foreach (var filePath in filesPaths)
        {
            // read the psmtsv psmFile
            var psmtsv = new PsmFromTsvFile(filePath);
            psmtsv.LoadResults();

            // make the calibration logger
            FileLoggers.Add(new FileLogger(psmtsv));
        }
    }

    public void Calibrate()
    {
        foreach (var fileLogger in FileLoggers)
        {
            fileLogger.Calibrate();
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