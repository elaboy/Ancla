using System.Diagnostics;
using AnchorLib;
using Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Plotly.NET;
using SharpLearning.Optimization;
using Chart = Plotly.NET.CSharp.Chart;

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

    [Test]
    public void TestDbConnectionInit()
    {
        string path = "testing_init.db";
        bool anyError = false;

        DbOperations.DbConnectionInit(path, out anyError);

        Assert.That(!anyError);
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
            new List<string>() { @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv" };

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

    #endregion


    #region One File at a time to show how CLT reflets on the collection of evidence

    /*
     * This method's purpose is to test how the variance for the same peptide keeps reducing on each
     * database insertion. Visualize how both scatter plots keeps increasing in linearity on each iteration.
     */
    [Test]
    public void TestIncrementalFileAndTestDistributions()
    {
        var psmFilePath = new List<string>()
        {
            @"D:\MannPeptideResults/A549_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/GAMG_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/HEK293_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/Hela_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/HepG2AllPSMs.psmtsv",
            @"D:\MannPeptideResults/Jurkat_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/LanCap_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/MCF7_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/RKO_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/U2OS_AllPSMs.psmtsv",
            @"D:\OtherPeptideResultsForTraining/JurkatMultiProtease_AllPSMs.psmtsv"
        };

        string dbPath = "IncrementalFilesAndDecreaseInDitribution";
        bool anyError = false;

        DbOperations.DbConnectionInit(dbPath, out anyError);

        // List of generic charts
        List<GenericChart.GenericChart> charts = new();

        for (int i = 0; i < 12; i++)
        {
            var psms = PsmService.GetPsms(new() { psmFilePath[i] });

            //remove psms whose file name is "12-18-17_frac3-calib-averaged"
            psms = psms.Where(p => p.FileName != "12-18-17_frac3-calib-averaged").ToList();

            var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
            optionsBuilder.UseSqlite(@"Data Source = " + dbPath);

            using (var context = new PsmContext(optionsBuilder.Options))
            {
                DbOperations.AnalizeAndAddPsmsBulk(context, psms);

                // Get the linear model
                var overlapsFromDatabase = DbOperations.GetFullSequencesOverlaps(context, psms);

                // Fit the linear model
                var linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase);

                // Transform the experimental retention times
                var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(
                    overlapsFromDatabase,
                    linearModel);

                // Plot the scatter plot
                GenericChart.GenericChart scatterPlot =
                    DbOperations.GetTransformationScatterPlot(transformedExperimental);

                // Add the scatter plot to the list of charts
                charts.Add(scatterPlot);
            }
        }

        // create a grid for all the charts
        var grid = Chart.Grid(charts.ToArray(), 12, 1);

        // show the grid
        GenericChartExtensions.Show(grid);
    }
    #endregion

    [Test]
    public void TestGetDistributions()
    {
        var psmFilePath = new List<string>()
        {
            @"D:\MannPeptideResults/A549_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/GAMG_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/HEK293_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/Hela_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/HepG2AllPSMs.psmtsv",
            @"D:\MannPeptideResults/Jurkat_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/LanCap_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/MCF7_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/RKO_AllPSMs.psmtsv",
            @"D:\MannPeptideResults/U2OS_AllPSMs.psmtsv",
        };

        string dbPath = "DistributionTest2";
        bool anyError = false;

        DbOperations.DbConnectionInit(dbPath, out anyError);

        var psms = PsmService.GetPsms(psmFilePath);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(@"Data Source = " + dbPath);

        List<GenericChart.GenericChart> charts = new();

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AnalizeAndAddPsmsBulk(context, psms);
        }

        string jurkatPath = @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv";

        var psmsJurkat = PsmService.GetPsms(new() { jurkatPath });

        //remove psms whose file name is "12-18-17_frac3-calib-averaged"
        psmsJurkat = psmsJurkat.Where(p => p.FileName == "12-18-17_frac3-calib-averaged").ToList();
        
        using (var context = new PsmContext(optionsBuilder.Options))
        {
            // Get the linear model
            var overlapsFromDatabase = DbOperations.GetFullSequencesOverlaps(context, psmsJurkat);

            // Fit the linear model
            var linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase);

            // Transform the experimental retention times
            var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(
                overlapsFromDatabase,
                linearModel);

            // Plot the distribution 
            charts.Add(DbOperations.GetDistributions(transformedExperimental));
        }

        //show the grid
        charts.First().Show();
    }
}