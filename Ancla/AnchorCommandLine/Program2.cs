using Proteomics.PSM;
using Readers;
using SQLitePCL;

namespace AnchorCommandLine;

public class Calibration
{

}

public class File
{
    public string Path { get; set; }
    public Dictionary<string, List<PsmFromTsv>> RawFiles { get; set; }
    public KeyValuePair<string, List<PsmFromTsv>> BaseRawFile { get; set; }

    public File(string path)
    {
        Path = path;

        // read the psmtsv file
        var psmtsvFile = new PsmFromTsvFile(path);
        psmtsvFile.LoadResults();

        // group them by FileName 
        var psmsGroupedByRawFileName = psmtsvFile
            .GroupBy(x => x.FileNameWithoutExtension);

        RawFiles = new();

        // ensemble the dictionary
        foreach (var rawFile in psmsGroupedByRawFileName)
        {
            RawFiles.Add(rawFile.Key, rawFile
                .Select(x => x)
                .Where(p => p.QValue <= 0.01 & 
                            p.PEP <= 0.5 & 
                            p.AmbiguityLevel == "1" & 
                            p.DecoyContamTarget == "T")
                .DistinctBy(seq => seq.FullSequence)
                .ToList());
}

        // set the base raw file
        BaseRawFile = RawFiles.First();
        RawFiles.Remove(BaseRawFile.Key);
    }

    public void PairWiseCalibration(KeyValuePair<string, List<PsmFromTsv>> rawFile)
    {
        var anchors = rawFile.Value.Intersect(BaseRawFile.Value);
    }
}

public class RawFile
{
    public string FileName { get; set; }
    public List<PsmFromTsv> Psms { get; set; }
    public List<FullSequence> FullSequences { get; set; }

    public RawFile(string fileName, List<PsmFromTsv> psmList)
    {
        FileName = fileName;
        FullSequences = new List<FullSequence>();
        Psms = new List<PsmFromTsv>();
    }



    private void ExtractFullSequences()
    {
        // Filter the sequences from the psms
        Psms = Psms.Where(p => p.QValue <= 0.01 &
                               p.PEP <= 0.5 &
                               p.AmbiguityLevel == "1" &
                               p.DecoyContamTarget == "T").ToList();

        // Make sure there are no duplicate Full Sequences, if there are, take the median of the retention times
        var fullSequencesGroups = Psms.GroupBy(p => p.FullSequence);

        foreach (var fullSequenceGroup in fullSequencesGroups)
        {
            var fullSequence = new FullSequence(fullSequenceGroup.Key);
            
            foreach (var psm in fullSequenceGroup)
            {
                fullSequence.FileNameRetentionTime.Add(FileName, psm.RetentionTime);
            }
            FullSequences.Add(new FullSequence(fullSequence.FullSeq){});
        }
    }
}

public class FullSequence
{
    public string FullSeq { get; set; }
    public Dictionary<string, double?> FileNameRetentionTime { get; set; }

    public FullSequence(string fullSeq)
    {
        FullSeq = fullSeq;
        FileNameRetentionTime = new Dictionary<string, double?>();
    }
}