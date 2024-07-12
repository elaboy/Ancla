using Easy.Common.Extensions;
using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Data;
using MzLibUtil;
using Omics.SpectrumMatch;
using Proteomics.PSM;
using Readers;

namespace AnchorCommandLine.RetentionTime;
public class Harmonizer
{
    public IEnumerable<IRetentionTimeHarmonizer> AllSpeciesInAllFiles { get; set; }

    public Dictionary<string, List<IRetentionTimeHarmonizer>> FilesInHarmonizer = new();

    public Dictionary<string, Dictionary<string, double>> HarmonizedSpecies = new();

    public Harmonizer(string path, out List<string> warnings)
    {
        AllSpeciesInAllFiles = PsmFromTsvReader.ReadTsv(path, out warnings);

        // group by FileName 
        var files = AllSpeciesInAllFiles
            .GroupBy(x => x.FileName);

        // populates FilesInHarmonizer (foreach FileName there is a List of IRetentionTimeHarmonizer
        foreach (var file in files)
        {
            FilesInHarmonizer.Add(file.Key, new List<IRetentionTimeHarmonizer>());
            FilesInHarmonizer[file.Key].AddRange(file.Select(x => x));
        }

        // order by count
        FilesInHarmonizer = FilesInHarmonizer.OrderByDescending(x => x.Value.Count)
            .ToDictionary(p => p.Key, p => p.Value);

        // Get all identifiers that are going to be harmonized and add them to HarmonizedSpecies
        var allSequencesPresent = AllSpeciesInAllFiles
            .DistinctBy(x => x.Identifier);

        //allSequencesPresent.ForEach(x => HarmonizedSpecies.Add(x.Identifier, new Dictionary<string, double>()));

        // Add all from the first file
        var firstLeader = FilesInHarmonizer.First();

        //// add the identifiers
        //firstLeader.Value.ForEach(x=> HarmonizedSpecies.Add(x.Identifier, new Dictionary<string, double>()));

        foreach (var identifier in firstLeader.Value)
        {
            if(HarmonizedSpecies.ContainsKey(identifier.Identifier))
                HarmonizedSpecies[identifier.Identifier].Add(firstLeader.Key, identifier.RetentionTime);
            else
            {
                HarmonizedSpecies.Add(identifier.Identifier, new Dictionary<string, double>());
                HarmonizedSpecies[identifier.Identifier].Add(firstLeader.Key, identifier.RetentionTime);

            }
        }

        // One iteration of PairwiseCalibration to set an initial calibration
        foreach (var file in FilesInHarmonizer.Where(x => !x.Key.Equals(firstLeader.Key)))
            InitialPairWiseCalibration(file.Key);
    }

    public void Calibrate(int epochs = 10, int minimumAnchors = 2)
    {
        for (int i = 0; i < epochs; i++)
        {
            foreach (var file in FilesInHarmonizer.Keys)
            {
                var anchorsAvailable = HarmonizedSpecies
                    .Where(x => x.Value.Count > minimumAnchors)
                    .Select(x => x.Key);

                // pop out the file to re-calibrate
                var toCalibrate = HarmonizedSpecies
                    .Where(x => x.Value.ContainsKey(file))
                    .ToDictionary(p => p.Key, p => p.Value);

                //removes the popped out file from the Harmonized Species
                HarmonizedSpecies.ForEach(x => x.Value.Remove(file));

                // get anchors
                Dictionary<string, (double anchorRetentionTime, double retentionTime)> anchors = anchorsAvailable
                    .Intersect(toCalibrate
                        .Select(x => x.Key))
                    .ToDictionary(x => x, x => (HarmonizedSpecies[x]
                .Select(x => x.Value).Median(), toCalibrate
                .Select(x => x.Value.First().Value).First()));

                // make the anchors PreCalibratedObjects
                List<PreCalibratedSequence> preCalibratedSequences = new();

                var predictionEngine = MakePipeline(anchors);

                foreach (var unCalibratedFollowerSpecies in toCalibrate)
                {
                    var prediction = predictionEngine.Predict(new PreCalibratedSequence()
                    {
                        FullSequence = unCalibratedFollowerSpecies.Key,
                        UnCalibratedRetentionTime = (float)unCalibratedFollowerSpecies.Value.First().Value
                    });

                    HarmonizedSpecies[unCalibratedFollowerSpecies.Key].Add(file, prediction.CalibratedRetentionTime);
                }
            }
        }
    }

    public PredictionEngine<PreCalibratedSequence, CalibratedSequence> MakePipeline(Dictionary<string, (double anchorRetentionTime, double retentionTime)> anchors)
    {
        MLContext mlContext = new MLContext();

        List<PreCalibratedSequence> preCalibratedSequences = new();

        // Prepare the data for the dataview
        foreach (var anchor in anchors)
        {
            preCalibratedSequences.Add(new PreCalibratedSequence()
            {
                FullSequence = anchor.Key,
                AnchorRetentionTime = (float)anchor.Value.anchorRetentionTime,
                UnCalibratedRetentionTime = (float)anchor.Value.retentionTime
            });
        }

        var dataView = mlContext.Data.LoadFromEnumerable(preCalibratedSequences.ToArray());

        // Make the model pipeline
        var pipeline = mlContext.Transforms
            .CopyColumns("Label", nameof(PreCalibratedSequence.AnchorRetentionTime))
            .Append(mlContext.Transforms.Concatenate("Features", nameof(PreCalibratedSequence.UnCalibratedRetentionTime)))
            .Append(mlContext.Regression.Trainers.Ols("Label", "Features"));

        // train the model
        var model = pipeline.Fit(dataView);

        // makes the prediction engine to predict the follower retention times
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PreCalibratedSequence, CalibratedSequence>(model);

        return predictionEngine;
    }

    private void InitialPairWiseCalibration(string followerFile)
    {
        Dictionary<string, (double median, double retentionTime)> anchors = HarmonizedSpecies.Keys
            .Intersect(FilesInHarmonizer[followerFile]
                .Select(x => x.Identifier))
            .ToDictionary(x => x, x => (HarmonizedSpecies[x]
                .Select(p => p.Value).Median(), FilesInHarmonizer[followerFile]
                .Select(e => e.RetentionTime).Median()));

        var intersect = HarmonizedSpecies.Keys.Intersect(FilesInHarmonizer[followerFile].Select(x => x.Identifier));

        Dictionary<string, (double median, )>

        var predictionEngine = MakePipeline(anchors);

        foreach (var unCalibratedFollowerSpecies in FilesInHarmonizer[followerFile])
        {
            var prediction = predictionEngine.Predict(new PreCalibratedSequence()
            {
                FullSequence = unCalibratedFollowerSpecies.Identifier,
                UnCalibratedRetentionTime = (float)unCalibratedFollowerSpecies.RetentionTime
            });
            if(HarmonizedSpecies.ContainsKey(unCalibratedFollowerSpecies.Identifier))
                HarmonizedSpecies[unCalibratedFollowerSpecies.Identifier].Add(followerFile, prediction.CalibratedRetentionTime);
            else
            {
                HarmonizedSpecies.Add(unCalibratedFollowerSpecies.Identifier, new Dictionary<string, double>());
                HarmonizedSpecies[unCalibratedFollowerSpecies.Identifier].Add(unCalibratedFollowerSpecies.FileName, prediction.CalibratedRetentionTime);
            }
        }
    }
}

public class Psm : PsmFromTsv, IRetentionTimeHarmonizer
{
    public Psm(string line, char[] split, Dictionary<string, int> parsedHeader) : base(line, split, parsedHeader)
    { }

    public string FileName
    {
        get => FileNameWithoutExtension;
        set => FileNameWithoutExtension = value;
    }

    public new double RetentionTime
    {
        get => base.RetentionTime!.Value;
        set => base.RetentionTime = value;
    }

    public string Identifier => FullSequence;
}

public class PreCalibratedSequence
{
    public string FullSequence { get; set; }
    public float AnchorRetentionTime { get; set; }
    public float UnCalibratedRetentionTime { get; set; }
}

public class CalibratedSequence
{
    [ColumnName("Score")]
    public float CalibratedRetentionTime { get; }
}
public static class PsmFromTsvReader
{
    /// <summary>
    /// Legacy method for reading PsmFromTsv files, creates a generic SpectrumMatchFromTsv object for each line
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="warnings"></param>
    /// <returns></returns>
    /// <exception cref="MzLibException"></exception>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static List<IRetentionTimeHarmonizer> ReadTsv(string filePath, out List<string> warnings)
    {
        List<SpectrumMatchFromTsv> psms = new List<SpectrumMatchFromTsv>();
        warnings = new List<string>();

        StreamReader reader = null;
        try
        {
            reader = new StreamReader(filePath);
        }
        catch (Exception e)
        {
            throw new MzLibException("Could not read file: " + e.Message);
        }

        int lineCount = 0;

        string line;
        Dictionary<string, int> parsedHeader = null;

        var fileType = filePath.ParseFileType();
        while (reader.Peek() > 0)
        {
            lineCount++;

            line = reader.ReadLine();

            if (lineCount == 1)
            {
                parsedHeader = ParseHeader(line);
                continue;
            }

            try
            {
                psms.Add(new Psm(line, Split, parsedHeader));
            }
            catch (Exception e)
            {
                warnings.Add("Could not read line: " + lineCount);
            }
        }

        reader.Close();

        if (lineCount - 1 != psms.Count)
        {
            warnings.Add("Warning: " + (lineCount - 1 - psms.Count) + " PSMs were not read.");
        }

        // filter the psms
        psms = psms.Where(p => p.QValue <= 0.01 &
                               p.PEP <= 0.5 &
                               p.AmbiguityLevel == "1" &
                               p.DecoyContamTarget == "T")
            .GroupBy(x => x.FileNameWithoutExtension)
            .Select(x => x
                .DistinctBy(x => x.FullSequence))
            .SelectMany(x => x).ToList();

        return psms.Cast<IRetentionTimeHarmonizer>().ToList();
    }

    private static readonly char[] Split = { '\t' };
    public static Dictionary<string, int> ParseHeader(string header)
    {
        var parsedHeader = new Dictionary<string, int>();
        var spl = header.Split(Split);

        parsedHeader.Add(SpectrumMatchFromTsvHeader.FullSequence, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.FullSequence));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.Ms2ScanNumber, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Ms2ScanNumber));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.FileName, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.FileName));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.TotalIonCurrent, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.TotalIonCurrent));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PrecursorScanNum, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PrecursorScanNum));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PrecursorCharge, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PrecursorCharge));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PrecursorMz, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PrecursorMz));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PrecursorMass, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PrecursorMass));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.Score, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Score));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.DeltaScore, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.DeltaScore));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.Notch, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Notch));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BaseSequence, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BaseSequence));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.EssentialSequence, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.EssentialSequence));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.AmbiguityLevel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.AmbiguityLevel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MissedCleavages, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MissedCleavages));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MassDiffDa, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MassDiffDa));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MassDiffPpm, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MassDiffPpm));

        //Handle legacy input
        if (spl.Contains(SpectrumMatchFromTsvHeader.Accession))
        {
            parsedHeader.Add(SpectrumMatchFromTsvHeader.SpectrumMatchCount, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.SpectrumMatchCount));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.MonoisotopicMass, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MonoisotopicMass));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Accession, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Accession));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Name, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Name));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Description, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Description));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.StartAndEndResiduesInFullSequence, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.StartAndEndResiduesInFullSequence));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.NextResidue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PreviousResidue));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.PreviousResidue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.NumExperimentalPeaks));
        }
        else
        {
            parsedHeader.Add(SpectrumMatchFromTsvHeader.SpectrumMatchCount, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PsmCount));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.MonoisotopicMass, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PeptideMonoMass));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Accession, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.ProteinAccession));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Name, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.ProteinName));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.Description, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PeptideDescription));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.StartAndEndResiduesInFullSequence, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.StartAndEndResiduesInProtein));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.NextResidue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PreviousAminoAcid));
            parsedHeader.Add(SpectrumMatchFromTsvHeader.PreviousResidue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.NextAminoAcid));
        }

        parsedHeader.Add(SpectrumMatchFromTsvHeader.GeneName, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.GeneName));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.OrganismName, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.OrganismName));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.IntersectingSequenceVariations, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.IntersectingSequenceVariations));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.IdentifiedSequenceVariations, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.IdentifiedSequenceVariations));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.SpliceSites, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.SpliceSites));

        parsedHeader.Add(SpectrumMatchFromTsvHeader.DecoyContaminantTarget, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.DecoyContaminantTarget));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MatchedIonMzRatios, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MatchedIonMzRatios));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MatchedIonIntensities, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MatchedIonIntensities));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.MatchedIonMassDiffDa, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.MatchedIonMassDiffDa));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.SpectralAngle, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.SpectralAngle));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.QValue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.QValue));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.QValueNotch, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.QValueNotch));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PEP, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PEP));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.PEP_QValue, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.PEP_QValue));

        parsedHeader.Add(SpectrumMatchFromTsvHeader.CrossTypeLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.CrossTypeLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.LinkResiduesLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.LinkResiduesLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.ProteinLinkSiteLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.ProteinLinkSiteLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.RankLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.RankLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideProteinAccessionLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideProteinAccessionLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideProteinLinkSiteLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideProteinLinkSiteLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideBaseSequenceLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideBaseSequenceLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideFullSequenceLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideFullSequenceLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideTheoreticalMassLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideTheoreticalMassLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideScoreLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideScoreLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideRankLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideRankLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideMatchedIonsLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideMatchedIonsLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.BetaPeptideMatchedIonIntensitiesLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.BetaPeptideMatchedIonIntensitiesLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.XLTotalScoreLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.XLTotalScoreLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.ParentIonsLabel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.ParentIonsLabel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.Ms2ScanRetentionTime, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.Ms2ScanRetentionTime));


        parsedHeader.Add(SpectrumMatchFromTsvHeader.GlycanMass, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.GlycanMass));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.GlycanStructure, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.GlycanStructure));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.GlycanComposition, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.GlycanComposition));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.GlycanLocalizationLevel, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.GlycanLocalizationLevel));
        parsedHeader.Add(SpectrumMatchFromTsvHeader.LocalizedGlycan, Array.IndexOf(spl, SpectrumMatchFromTsvHeader.LocalizedGlycan));

        return parsedHeader;
    }
}