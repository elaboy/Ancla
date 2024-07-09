using Microsoft.ML;
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
        rawFile.Value.OrderBy(x => x.RetentionTime);
        BaseRawFile.Value.OrderBy(x => x.RetentionTime);

        var anchors = rawFile.Value.Intersect(BaseRawFile.Value);

        var sequences = new List<string>();
        var baseRt = new List<double>();
        var followerRt = new List<double>();

        foreach (var anchor in anchors)
        {
            // get the retention time of the anchor in the base raw file
            var baseRawFileRetentionTime = BaseRawFile.Value
                .Where(x => x.FullSequence == anchor.FullSequence)
                .Select(x => x.RetentionTime)
                .FirstOrDefault();

            // get the retention time of the anchor in the raw file
            var rawFileRetentionTime = rawFile.Value
                .Where(x => x.FullSequence == anchor.FullSequence)
                .Select(x => x.RetentionTime)
                .FirstOrDefault();

            sequences.Add(anchor.FullSequence);
            baseRt.Add(baseRawFileRetentionTime.Value);
            followerRt.Add(rawFileRetentionTime.Value);
        }

        // use ml.net to train a linear regression model using the leader and follower retention times as training data
        MLContext mlContext = new MLContext();
        var data = new List<Anchor>();

        for (var i = 0; i < baseRt.Count; i++)
        {
            data.Add(new Anchor
            {
                FullSequence = sequences[i],
                LeaderRetentionTime = (float)baseRt[i],
                FollowerRetentionTime = (float)followerRt[i]
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

        foreach (var fullSequence in rawFile.Value)
        {
            var prediction = predictionEngine.Predict(new Anchor
            {
                FullSequence = fullSequence.FullSequence,
                FollowerRetentionTime = (float)fullSequence.RetentionTime.Value
            });
        }

        foreach(var psm in rawFile.Value)
        {

        }
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