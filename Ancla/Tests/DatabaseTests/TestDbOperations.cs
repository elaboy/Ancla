using System.Diagnostics;
using AnchorLib;
using Database;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.VisualStudio.TestPlatform.Common;
using Plotly.NET;
using ScottPlot;
using ScottPlot.DataSources;
using ScottPlot.Plottables;
using SharpLearning.Optimization;
using Chart = Plotly.NET.CSharp.Chart;
using Color = ScottPlot.Color;
using Histogram = ScottPlot.Statistics.Histogram;

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

    [Test]
    public void TestScottPlotScatter()
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
        };

        string dbPath = @"D:\\ScottPlotTesting1.db";
        bool anyError = false;

        DbOperations.DbConnectionInit(dbPath, out anyError);

        var jurkatPath = new List<string>() { @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPeptides.psmtsv" };

        //var psms = PsmService.GetPsms(psmFilePath);

        
        var jurkatPsms = PsmService.GetPsms(jurkatPath);

        //remove psms whose file name is "12-18-17_frac3-calib-averaged"
        //jurkatPsms= jurkatPsms.Where(p => p.FileName == "12-18-17_frac6-calib-averaged").ToList();

        var setsFromJurkat = jurkatPsms.GroupBy(x => x.FileName);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(@"Data Source = " + dbPath);

        //using (var context = new PsmContext(optionsBuilder.Options))
        //{
        //    DbOperations.AnalizeAndAddPsmsBulk(context, psms);
        //}

        foreach(var file in setsFromJurkat)
        {
            using (var context = new PsmContext(optionsBuilder.Options))
            {
                //Get files psms 
                var filePsms = file.ToList();

                // Get the linear model
                var overlapsFromDatabase = DbOperations.GetFullSequencesOverlaps(context, filePsms);

                // Fit the linear model
                try
                {
                    var linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase);
                    // Transform the experimental retention times
                    var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(
                        overlapsFromDatabase, linearModel);

                    // Plot the scatter plot

                    ScottPlot.Plot plt = new();

                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item2.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.DodgerBlue));

                    // add R2

                    var preTransformation = GoodnessOfFit.CoefficientOfDetermination(
                        transformedExperimental.Select(x => x.Item2.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray());


                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item3.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.DarkRed));



                    // add R2
                    var postTransformation = GoodnessOfFit.CoefficientOfDetermination(
                        transformedExperimental.Select(x => x.Item3.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray());


                    // add both R2 values
                    plt.Title("Pre-Transformation R2: " + preTransformation.Round(4).ToString() + "| Post-Transformation R2: " +
                              postTransformation.Round(4).ToString());

                    // plot db vs db 
                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.ForestGreen));

                    plt.SavePng(@"D:\transformationFor" + file.Key + ".png", 800, 400);

                    // histogram with z scores 

                }
                catch (Exception e)
                {
                    continue;
                }
            }
        }
    }

    [Test]
    public void TestScottPlotScatterAndDistributions()
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
        };

        string dbPath = @"D:\\ScottPlotTesting1.db";
        bool anyError = false;

        DbOperations.DbConnectionInit(dbPath, out anyError);

        var jurkatPath = new List<string>() { @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPeptides.psmtsv" };

        //var psms = PsmService.GetPsms(psmFilePath);


        var jurkatPsms = PsmService.GetPsms(jurkatPath);

        //remove psms whose file name is "12-18-17_frac3-calib-averaged"
        //jurkatPsms= jurkatPsms.Where(p => p.FileName == "12-18-17_frac6-calib-averaged").ToList();

        var setsFromJurkat = jurkatPsms.GroupBy(x => x.FileName);

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseSqlite(@"Data Source = " + dbPath);

        //using (var context = new PsmContext(optionsBuilder.Options))
        //{
        //    DbOperations.AnalizeAndAddPsmsBulk(context, psms);
        //}

        foreach (var file in setsFromJurkat)
        {
            using (var context = new PsmContext(optionsBuilder.Options))
            {
                //Get files psms 
                var filePsms = file.ToList();

                // Get the linear model
                var overlapsFromDatabase = DbOperations.GetFullSequencesOverlaps(context, filePsms);

                // Fit the linear model
                try
                {
                    var linearModel = DbOperations.FitLinearModelToData(overlapsFromDatabase);
                    // Transform the experimental retention times
                    var transformedExperimental = DbOperations.TransformExperimentalRetentionTimes(
                        overlapsFromDatabase, linearModel);

                    // Plot the scatter plot

                    ScottPlot.Plot plt = new();

                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item2.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.DodgerBlue));

                    // add R2

                    var preTransformation = GoodnessOfFit.CoefficientOfDetermination(
                        transformedExperimental.Select(x => x.Item2.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray());


                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item3.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.DarkRed));



                    // add R2
                    var postTransformation = GoodnessOfFit.CoefficientOfDetermination(
                        transformedExperimental.Select(x => x.Item3.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray());


                    // add both R2 values
                    plt.Title("Pre-Transformation R2: " + preTransformation.Round(4).ToString() +
                              "| Post-Transformation R2: " +
                              postTransformation.Round(4).ToString());

                    // plot db vs db 
                    plt.Add.ScatterPoints(transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray(),
                        transformedExperimental.Select(y => y.Item1.ScanRetentionTime).ToArray(),
                        color: Color.FromColor(System.Drawing.Color.ForestGreen));

                    plt.SavePng(@"D:\transformationFor" + file.Key + ".png", 800, 400);

                    // Calculate the standard deviations of the database and the experimental data
                    var dbStdDev = transformedExperimental.Select(x => x.Item1.ScanRetentionTime).ToArray()
                        .StandardDeviation();
                    var expStdDev = transformedExperimental.Select(x => x.Item2.ScanRetentionTime).ToArray()
                        .StandardDeviation();
                    var postStdDev = transformedExperimental.Select(x => x.Item3.ScanRetentionTime).ToArray()
                        .StandardDeviation();

                    // Calculate the z scores
                    var dbZScores = transformedExperimental
                        .Select(x => (x.Item1.ScanRetentionTime - dbStdDev) / dbStdDev).ToArray();

                    var expZScores = transformedExperimental
                        .Select(x => (x.Item2.ScanRetentionTime - expStdDev) / expStdDev).ToArray();

                    var postZScores = transformedExperimental
                        .Select(x => (x.Item3.ScanRetentionTime - postStdDev) / postStdDev).ToArray();

                    // Create a histogram for the z scores
                    ScottPlot.Plot plt2 = new();

                    var zScores = DbOperations.GetZscores(transformedExperimental);

                    ScottPlot.Statistics.Histogram histPre =
                        new Histogram(zScores.Item1.Min(), zScores.Item1.Max(), 100);

                    ScottPlot.Statistics.Histogram histPost =
                        new Histogram(zScores.Item2.Min(), zScores.Item2.Max(), 100);

                    histPre.AddRange(zScores.Item1);
                    histPost.AddRange(zScores.Item2);

                    plt2.Add.Bars(histPre.Bins, histPre.GetNormalized());

                    plt2.Add.Bars(histPost.Bins, histPost.GetNormalized());

                    //plt2.Add.Boxes(boxPlot);
                    plt2.SavePng(@"D:\zScoresFor" + file.Key + ".png", 1200, 800);
                }
                catch (Exception e)
                {
                    continue;
                }
            }
        }


    }

    [Test]
    public void TestSaveAsCSV()
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

        string dbPath = @"D:\DistributionTest2";
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

            // Save data as CSV
            DbOperations.SaveAsCSV(transformedExperimental, @"D:\transformedData.csv");
        }
    }

    [Test]
    public void TestSaveAsCSVRAW()
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

        string dbPath = @"D:\toCSVRAW.db";
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

        //string jurkatPath = @"D:\OtherPeptideResultsForTraining\JurkatMultiProtease_AllPSMs.psmtsv";

        //var psmsJurkat = PsmService.GetPsms(new() { jurkatPath });

        ////remove psms whose file name is "12-18-17_frac3-calib-averaged"
        //psmsJurkat = psmsJurkat.Where(p => p.FileName == "12-18-17_frac3-calib-averaged").ToList();

        //using (var context = new PsmContext(optionsBuilder.Options))
        //{
        //    // Get the linear model
        //    var overlapsFromDatabase = DbOperations.GetFullSequencesOverlaps(context, psmsJurkat);
        //    // Save data as CSV
        //    //DbOperations.SaveAsCSV(overlapsFromDatabase, @"D:\transformedData_RAW.csv");
        //}
    }
}