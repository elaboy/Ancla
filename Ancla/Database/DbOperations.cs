using AnchorLib;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Plotly.NET;
using Plotly.NET.CSharp;
using Plotly.NET.ImageExport;
using Plotly.NET.LayoutObjects;
using Plotly.NET.TraceObjects;
using ThermoFisher.CommonCore.Data;
using Chart = Plotly.NET.CSharp.Chart;
using GenericChartExtensions = Plotly.NET.GenericChartExtensions;

namespace Database;

public static class DbOperations
{
    public static string ConnectionString =
        @"Data Source = D:\anchor_testing_linear_model.db";

    public static void DbConnectionInit(string dbPathAndName, out bool anyErrors)
    {
        // create a service collection and configure it for DbContext
        var services = new ServiceCollection()
            .AddDbContextFactory<PsmContext>(options =>
                options.UseSqlite(@"Data Source = " + dbPathAndName));

        // Builds the service provider
        var serviceProvider = services.BuildServiceProvider();

        // Get the DbContext factory from the service provider
        var dbContextFactory =
            serviceProvider.GetRequiredService<IDbContextFactory<PsmContext>>();

        // Creates new instance of the DbContext
        using (var dbContext = dbContextFactory.CreateDbContext())
        {
            try
            {
                // Apply any pending migrations
                dbContext.Database.Migrate();
                anyErrors = false;
                Console.WriteLine("Datbase migration successful.");
            }
            catch (Exception exception)
            {
                anyErrors = true;
                Console.WriteLine(exception.Message);
            }
        }
    }
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
                                     p.FullSequence == group.Key && 
                                     p.QValue <= 0.01);

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
        var psmsInDb = context.PSMs.ToList();

        List<PSM> psmsToUpload = new List<PSM>();

        // empty database, dont check for redundancy, else upload every psm with qvalue <= 0.01
        if (psmsInDb.IsNullOrEmpty())
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
                    var existingData = psmsInDb
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

    public static List<(PSM, PSM)> GetFullSequencesOverlaps(PsmContext context,
        List<PSM> psms)
    {

        var databasePsms = context.PSMs.ToList();

        List<(PSM, PSM)> overlappingPsms = new List<(PSM, PSM)>();

        Parallel.ForEach(psms, psm =>
        {
            var existingData = databasePsms
                .FirstOrDefault(p => p.FullSequence == psm.FullSequence);

            if (existingData != null)
            {
                overlappingPsms.Add((existingData, psm));
            }
        });

        return overlappingPsms;
    }

    #region Linear Regression Code

    public static (double, double) FitLinearModelToData(List<(PSM, PSM)> overlaps)
    {
        //database psms
        var y = overlaps.Select(psmTuple => psmTuple.Item1)
            .OrderByDescending(p => p.ScanRetentionTime)
            .ToArray();

        //experimental psms
        var x = overlaps.Select(psmTuple => psmTuple.Item2)
            .OrderByDescending(p => p.ScanRetentionTime)
            .ToArray();

        (double, double) model = Fit.Line(x.Select(p => p.ScanRetentionTime).ToArray(),
            y.Select(p => p.ScanRetentionTime).ToArray());

        var intercept = model.Item1;
        var slope = model.Item2;

        return (intercept, slope);
    }

    public static List<(PSM, PSM, PSM)> TransformExperimentalRetentionTimes(List<(PSM, PSM)> overlaps, (double, double) model)
    {
        var intercept = model.Item1;
        var slope = model.Item2;

        List<(PSM, PSM, PSM)> transformedData = new List<(PSM, PSM, PSM)>();

        foreach (var psm in overlaps)
        {
            transformedData.Add((psm.Item1, psm.Item2, new PSM()
            {
                FileName = psm.Item2.FileName,
                BaseSequence = psm.Item2.BaseSequence,
                FullSequence = psm.Item2.FullSequence,
                ScanRetentionTime = (psm.Item2.ScanRetentionTime * slope) + intercept,
                QValue = psm.Item2.QValue,
                PEP = psm.Item2.PEP,
                PEPQvalue = psm.Item2.PEPQvalue,
                PrecursorCharge = psm.Item2.PrecursorCharge,
                PrecursorMZ = psm.Item2.PrecursorMZ,
                PrecursorMass = psm.Item2.PrecursorMass,
                ProteinAccession = psm.Item2.ProteinAccession,
                ProteinName = psm.Item2.ProteinName,
                GeneName = psm.Item2.GeneName,
                OrganismName = psm.Item2.OrganismName,
                StartAndEndResidueInProtein = psm.Item2.StartAndEndResidueInProtein,
                MassErrorDaltons = psm.Item2.MassErrorDaltons,
                Score = psm.Item2.Score,
                TotalIonCurrent = psm.Item2.TotalIonCurrent,
                Notch = psm.Item2.Notch,
                AmbiguityLevel = psm.Item2.AmbiguityLevel,
                PeptideMonoisotopicMass = psm.Item2.PeptideMonoisotopicMass
            }));
        }

        return transformedData;
    }

    public static void TransformationScatterPlot(List<(PSM, PSM, PSM)> data)
    {
        //calculate R^2 value
        var pre_rSquared = GoodnessOfFit.RSquared(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item2.ScanRetentionTime).ToArray());

        var post_rSquared = GoodnessOfFit.RSquared(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item3.ScanRetentionTime).ToArray());

        var preTransformation = Chart.Scatter<double, double, string>(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item2.ScanRetentionTime).ToArray(),
            StyleParam.Mode.Markers, pre_rSquared.ToString());

        var postTransformation = Chart.Scatter<double, double, string>(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item3.ScanRetentionTime).ToArray(),
            StyleParam.Mode.Markers, post_rSquared.ToString());

        // make the two scatters into the same image using a grid
        var grid = Chart.Grid(new[] { preTransformation, postTransformation }, 2, 1);

        //remove timeout from puppeteer
        PuppeteerSharpRendererOptions.launchOptions.Timeout = 0;

        // save the plot
        //grid.SavePNG(@"D:\transformation_scatter_plot.png", EngineType: null, 600, 400);
        //show plot grid
        GenericChartExtensions.Show(grid);
    }

    public static GenericChart.GenericChart GetTransformationScatterPlot(List<(PSM, PSM, PSM)> data)
    {
        //calculate R^2 value
        var pre_rSquared = GoodnessOfFit.RSquared(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item2.ScanRetentionTime).ToArray());

        var post_rSquared = GoodnessOfFit.RSquared(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item3.ScanRetentionTime).ToArray());

        var preTransformation = Chart.Scatter<double, double, string>(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item2.ScanRetentionTime).ToArray(),
            StyleParam.Mode.Markers, pre_rSquared.ToString());

        var postTransformation = Chart.Scatter<double, double, string>(
            data.Select(d => d.Item1.ScanRetentionTime).ToArray(),
            data.Select(d => d.Item3.ScanRetentionTime).ToArray(),
            StyleParam.Mode.Markers, post_rSquared.ToString());

        // make the two scatters into the same image using a grid
        var grid = Chart.Grid(new[] { preTransformation, postTransformation }, 2, 1);

        //remove timeout from puppeteer
        PuppeteerSharpRendererOptions.launchOptions.Timeout = 0;

        return grid;
        // save the plot
        //grid.SavePNG(@"D:\transformation_scatter_plot.png", EngineType: null, 600, 400);
        //show plot grid
    }

    public static GenericChart.GenericChart GetDistributions(List<(PSM, PSM, PSM)> data)
    {
        //get residuals from database vs experimental 
        var preResiduals = data.Select(d => d.Item1.ScanRetentionTime - d.Item2.ScanRetentionTime).ToArray();

        //get residuals from database vs transformedExperimental

        var postResiduals = data.Select(d => d.Item1.ScanRetentionTime - d.Item3.ScanRetentionTime).ToArray();

        //histogram with residuals
        var preResidualsHistogram = Chart.Histogram<double, double, string>(preResiduals, HistNorm:new Optional<StyleParam.HistNorm>(StyleParam.HistNorm.Density, false));

        //histogram with residuals
        var postResidualsHistogram = Chart.Histogram<double, double, string>(postResiduals, HistNorm: new Optional<StyleParam.HistNorm>(StyleParam.HistNorm.Density, false));

        // make the two histograms into the same image using a grid
        var grid = Chart.Grid(new[] { preResidualsHistogram, postResidualsHistogram }, 1, 2);

        return grid;
    }

    #endregion
}

