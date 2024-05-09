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


}