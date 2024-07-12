using Proteomics.PSM;

namespace AnchorCommandLine;

public class RawFileLogger
{
    public string RawFileName { get; set; }
    public IEnumerable<PsmFromTsv> Psms { get; set; }
    public Dictionary<string, double> FullSequenceWithScanRetentionTime = new();
    public RawFileLogger(string rawFileName, IEnumerable<PsmFromTsv> psms)
    {
        RawFileName = rawFileName;
        // TODO Filter the psms
        Psms = psms.Where(p => p.QValue <= 0.01 &
                               p.PEP <= 0.5 &
                               p.AmbiguityLevel == "1" &
                               p.DecoyContamTarget == "T").ToList();

        // Get the median retention time for each full sequence that are repeated in the raw psmFile
        var fullSequences = Psms
            .GroupBy(p => p.FullSequence);
        foreach (var fullSequence in fullSequences)
        {
            FullSequenceWithScanRetentionTime
                .Add(fullSequence.Key, fullSequence
                    .Select(p => p.RetentionTime.Value).First());
        }
    }
}