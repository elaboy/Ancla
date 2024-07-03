using Database;
using Easy.Common.Interfaces;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
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
    public Dictionary<string, RawFileLogger> RawFiles = new Dictionary<string, RawFileLogger>();
    public Dictionary<string, double?> FullSequencesPresentInFile = new Dictionary<string, double?>();
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

        FullSequencesPresentInFile = fullSequences.ToDictionary(p => p, p => (double?)null);
    }
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
                               p.PEP < 0.5 & 
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