using AnchorLib;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using ThermoFisher.CommonCore.Data;

namespace Database;

public static class DbOperations
{
    public static string ConnectionString = @"Data Source = D:\anchor_testing_linear_model.db";
    public static void AddPsms(PsmContext context, List<PSM> psms)
    {
        foreach (var psm in psms)
        {
            context.Add(psm);
        }
        context.SaveChanges();
    }

    public static void AddPsm(PsmContext context, PSM psm)
    {
        context.Add(psm);
        context.SaveChanges();
    }

    public static void AnalizeAndAddPsmEntry(PsmContext context, PSM psm)
    {
        // search for the psm FullSequence in the database
        var existingData = context.PSMs
            .FirstOrDefault(p => p.FileName == psm.FileName &&
                                 p.FullSequence == psm.FullSequence);

        // if nothing, upload it to the database
        if (existingData == null)
        {
            context.Add(psm);
            context.SaveChanges();
        }
    }

    public static void AnalizeAndAddPsmsFile(PsmContext context, List<PSM> psms)
    {
        foreach (var psm in psms)
        {
            AnalizeAndAddPsmEntry(context, psm);
        }
    }

    public static void AnalizeAndAddPsmsBulk(PsmContext context, List<PSM> psms)
    {

        //group all psms by full sequence 
        var groupedPsms = psms.GroupBy(p => p.FullSequence).ToList();

        var psmsToUpload = new List<PSM>();

        //take the median of the retention times
        foreach (var group in groupedPsms)
        {
            var medianRetentionTime = group.Select(p => p.ScanRetentionTime).Median();

            //search for the psm FullSequence in the database
            var existingData = context.PSMs
                .FirstOrDefault(p => p.FileName == group.First().FileName &&
                                     p.FullSequence == group.Key);

            //if nothing, upload it to the database
            if (existingData == null)
            {
                psmsToUpload.Add(new PSM
                {
                    FileName = group.First().FileName,
                    BaseSequence = group.First().BaseSequence,
                    FullSequence = group.Key,
                    ScanRetentionTime = medianRetentionTime,
                    QValue = group.First().QValue,
                    PEP = group.First().PEP,
                    PEPQvalue = group.First().PEPQvalue,
                    PrecursorCharge = group.First().PrecursorCharge,
                    PrecursorMZ = group.First().PrecursorMZ,
                    PrecursorMass = group.First().PrecursorMass,
                    ProteinAccession = group.First().ProteinAccession,
                    ProteinName = group.First().ProteinName,
                    GeneName = group.First().GeneName,
                    OrganismName = group.First().OrganismName,
                    StartAndEndResidueInProtein = group.First().StartAndEndResidueInProtein,
                    MassErrorDaltons = group.First().MassErrorDaltons,
                    Score = group.First().Score,
                    TotalIonCurrent = group.First().TotalIonCurrent,
                    Notch = group.First().Notch,
                    AmbiguityLevel = group.First().AmbiguityLevel,
                    PeptideMonoisotopicMass = group.First().PeptideMonoisotopicMass
                });
            }
        }

        // transaction to add all new psms to the database
        using (var transaction = context.Database.BeginTransaction())
        {

            // Add all new PSMs to the database
            context.AddRange(psmsToUpload);
            context.SaveChanges();
            transaction.Commit();
        }
    }

    public static void AddPsmsNonRedundant(PsmContext context, List<PSM> psms)
    {
        //one bulk transaction instead of multiple transactions (per psm) 
        var psmsInDB = context.PSMs.ToList();

        List<PSM> psmsToUpload = new List<PSM>();

        // empty database, dont check for redundancy, else upload every psm with qvalue <= 0.01
        if (psmsInDB.IsNullOrEmpty())
        {
            psmsToUpload.AddRange(psms.Where(p => p.QValue <= 0.01));
        }
        else
        {

            Parallel.ForEach(psms, psm =>
            {
                if (psm.QValue <= 0.01)
                {
                    // Fetch existing data in bulk
                    var existingData = psmsInDB
                        .FirstOrDefault(p => p.FileName == psm.FileName &&
                                             p.FullSequence == psm.FullSequence);
                    //.ToList();

                    if (existingData == null)
                    {
                        // Add new PSM if not found in the database
                        psmsToUpload.Add(psm);
                    }
                }

            });
        }

        // transaction to add all new psms to the database
        using (var transaction = context.Database.BeginTransaction())
        {

            // Add all new PSMs to the database
            context.AddRange(psmsToUpload);
            context.SaveChanges();
            transaction.Commit();
        }
    }

    public static List<PSM> FetchAnchors(PsmContext context, bool orderedByElution = false)
    {
        if (orderedByElution)
        {
            var anchors = context.PSMs.ToList();

            // group anchors by full sequence and average the retention times
            var groupedAnchors = anchors.GroupBy(a => a.FullSequence)
                .Select(g => new PSM
                {
                    FullSequence = g.Key,
                    ScanRetentionTime = g.Average(a => a.ScanRetentionTime)
                }).ToList();

            return groupedAnchors
                .OrderBy(p => p.ScanRetentionTime)
                .ToList();
        }
        else
        {
            var anchors = context.PSMs.ToList();

            // group anchors by full sequence and average the retention times
            var groupedAnchors = anchors.GroupBy(a => a.FullSequence)
                .Select(g => new PSM
                {
                    FullSequence = g.Key,
                    ScanRetentionTime = g.Average(a => a.ScanRetentionTime)
                }).ToList();

            return groupedAnchors;
        }
    }

    public static List<PSM> GetFullSequencesOverlaps(PsmContext context, List<PSM> psms)
    {
        
        var databasePsms = context.PSMs.ToList();

        List<PSM> overlappingPsms = new List<PSM>();

        Parallel.ForEach(psms, psm =>
        {
            var existingData = databasePsms
                .FirstOrDefault(p => p.FullSequence == psm.FullSequence);

            if (existingData != null)
            {
                overlappingPsms.Add(existingData);
            }
            else
            {
                //tag that psm to remove it from the list later
                psm.ScanRetentionTime = -999;
            }
        });

        return overlappingPsms;
    }

    #region Linear Regression Code

    public static (double, double) FitLinearModelToData(List<PSM> databasePsms, List<PSM> experimentalData)
    {
            var x = experimentalData
                .OrderByDescending(p => p.ScanRetentionTime)
                .ToArray();

            var y = databasePsms
                .OrderByDescending(p => p.ScanRetentionTime)
                .ToArray();

            (double, double) model = Fit.Line(x.Select(p => p.ScanRetentionTime).ToArray(),
                y.Select(p => p.ScanRetentionTime).ToArray());

            var intercept = model.Item1;
            var slope = model.Item2;

            return (intercept, slope);
    }

    public static List<PSM> TransformExperimentalRetentionTimes(List<PSM> experimentalData, (double, double) model)
    {
            var intercept = model.Item1;
            var slope = model.Item2;

            List<PSM> transformedData = new List<PSM>();

            foreach (var psm in experimentalData)
            {
                transformedData.Add(new PSM()
                {
                    FileName = psm.FileName,
                    BaseSequence = psm.BaseSequence,
                    FullSequence = psm.FullSequence,
                    ScanRetentionTime = (psm.ScanRetentionTime * slope) + intercept,
                    QValue = psm.QValue,
                    PEP = psm.PEP,
                    PEPQvalue = psm.PEPQvalue,
                    PrecursorCharge = psm.PrecursorCharge,
                    PrecursorMZ = psm.PrecursorMZ,
                    PrecursorMass = psm.PrecursorMass,
                    ProteinAccession = psm.ProteinAccession,
                    ProteinName = psm.ProteinName,
                    GeneName = psm.GeneName,
                    OrganismName = psm.OrganismName,
                    StartAndEndResidueInProtein = psm.StartAndEndResidueInProtein,
                    MassErrorDaltons = psm.MassErrorDaltons,
                    Score = psm.Score,
                    TotalIonCurrent = psm.TotalIonCurrent,
                    Notch = psm.Notch,
                    AmbiguityLevel = psm.AmbiguityLevel,
                    PeptideMonoisotopicMass = psm.PeptideMonoisotopicMass
                });
            }

            return experimentalData;
    }

    #endregion
}

public class ArchorOperations
{
    public void IdentifyAnchors()
    {
    }


}


