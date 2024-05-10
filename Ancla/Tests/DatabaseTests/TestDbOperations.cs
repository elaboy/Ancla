using AnchorLib;
using Database;
using Microsoft.EntityFrameworkCore;

namespace Tests.DatabaseTests;
public class TestDbOperations
{
    [Test]
    public void TestAddPsm()
    {
        var psmFilePath = new List<string>()
        {
            Path.Combine(TestContext.CurrentContext.TestDirectory, "ExcelEditedPeptide.psmtsv")
        };
        var psm = PsmService.GetPsms(psmFilePath);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AddPsm(context, psm[0]);
        }
    }

    [Test]
    public void TestAddPsms()
    {
        var psmFilePath = new List<string>()
        {
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\A549_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\GAMG_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HEK293_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Hela_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HepG2AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Jurkat_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\LanCap_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\MCF7_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\RKO_AllPeptides.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\U2OS_AllPeptides.psmtsv",

        };
        var psms = PsmService.GetPsms(psmFilePath);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AddPsms(context, psms);
        }
    }

    [Test]
    public void TestAddPsmsNonRedundant()
    {
        var psmFilePath = new List<string>()
        {
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\A549_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\GAMG_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HEK293_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Hela_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HepG2AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Jurkat_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\LanCap_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\MCF7_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\RKO_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\U2OS_AllPSMs.psmtsv",

        };



        var psms = PsmService.GetPsms(psmFilePath);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AddPsmsNonRedundant(context, psms);
        }
    }

    [Test]
    public void TestAnalizeAndAddPsmsBulk()
    {
        var psmFilePath = new List<string>()
        {
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\A549_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\GAMG_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HEK293_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Hela_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\HepG2AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\Jurkat_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\LanCap_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\MCF7_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\RKO_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\MannPeptideResults\U2OS_AllPSMs.psmtsv",
            @"\\Edwin-ml\F\RetentionTimeProject\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv"
        };

        var psms = PsmService.GetPsms(psmFilePath);

        //remove psms whose file name is "12-18-17_frac3-calib-averaged"
        psms = psms.Where(p => p.FileName != "12-18-17_frac3-calib-averaged").ToList();

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AnalizeAndAddPsmsBulk(context, psms);
        }
    }

    [Test]
    public void TestFetchAnchors()
    {
        using (var context = new PsmContext())
        {
            var anchors = DbOperations.FetchAnchors(context, true);
            Assert.That(anchors.Count > 0);
        }
    }

    #region Linear Regression Tests

    [Test]
    public void BulkDataLoad()
    {
        var psmFilePath = new List<string>()
        {
            @"D:\MannPeptideResults\A549_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\GAMG_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\HEK293_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\Hela_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\HepG2AllPSMs.psmtsv",
            @"D:\MannPeptideResults\Jurkat_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\LanCap_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\MCF7_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\RKO_AllPSMs.psmtsv",
            @"D:\MannPeptideResults\U2OS_AllPSMs.psmtsv",
            @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv"
        };

        var psms = PsmService.GetPsms(psmFilePath);

        //remove psms whose file name is "12-18-17_frac3-calib-averaged"
        psms = psms.Where(p => p.FileName != "12-18-17_frac3-calib-averaged").ToList();

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AnalizeAndAddPsmsBulk(context, psms);
        }
    }

    [Test]
    public void TestLinearRegressionN100()
    {
        List<string> experimental =
            new List<string>(){@"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv"};

        var psms = PsmService.GetPsms(experimental)
            .Where(p => p.FileName != "12-18-17_frac3-calib-averaged")
            //.Take(5000)
            .ToList();

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(DbOperations.ConnectionString);

        List<(PSM, PSM)> overlapsFromDatabase = new List<(PSM, PSM)>();

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            var overlaps = DbOperations.GetFullSequencesOverlaps(context, psms);
            overlapsFromDatabase.AddRange(overlaps);
        }

        (double, double) linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase);


        var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(
            overlapsFromDatabase,
            linearModel);

        DbOperations.TransformationScatterPlot(transformedExperimental);
    }

    //[Test]
    //public void TestLinearRegressionFull()
    //{
    //    List<string> experimental =
    //        new List<string>() { @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv" };

    //    var psms = PsmService.GetPsms(experimental)
    //        .Where(p => p.FileName != "12-18-17_frac3-calib-averaged" && p.QValue <= 0.01)
    //        //.Take(100)
    //        .ToList();

    //    var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
    //    optionsBuilder.UseSqlite(DbOperations.ConnectionString);

    //    List<PSM> overlapsFromDatabase = new List<PSM>();

    //    using (var context = new PsmContext(optionsBuilder.Options))
    //    {
    //        var overlaps = DbOperations.GetFullSequencesOverlaps(context, psms);
    //        overlapsFromDatabase.AddRange(overlaps);
    //    }

    //    //take out of psms the ones that are not in the database
    //    psms = psms.Where(p => overlapsFromDatabase.Any(o => o.ScanRetentionTime  != -999)).ToList();

    //    (double, double) linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase, psms);

    //    var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(psms, linearModel);


    //    //make table of the results where the first column is the full sequence of the full sequence from both the experimental and the database. 
    //    // The second column is the retention time from the experimental and the third column is the retention time from the database.
    //    // the third is the transformed retention time from the experimental

    //    var table = new List<(string, double, double, double)>();

    //    foreach (var psm in psms)
    //    {
    //        var databasePsm = overlapsFromDatabase.FirstOrDefault(p => p.FullSequence == psm.FullSequence);
    //        var transformedRetentionTime = transformedExperimental.FirstOrDefault(p => p.FullSequence == psm.FullSequence);
    //        table.Add((psm.FullSequence, psm.ScanRetentionTime, databasePsm.ScanRetentionTime, transformedRetentionTime.ScanRetentionTime));
    //    }

    //    Assert.That(table.Count > 0);
    //}

    #endregion

}