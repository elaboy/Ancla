using System.Runtime.InteropServices.ComTypes;
using AnchorLib;
using MathNet.Numerics.Statistics;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Transcriptomics;

namespace Database;
public static class Transactions
{
    public static void InitiateConnection(string databasePath)
    {
        // Creates a service collection and registers the database context
        var services = new ServiceCollection()
            .AddDbContextFactory<PsmContext>(options =>
            {
                options.UseSqlite($"Data Source={databasePath}");
            });

        // Builds the service provider
        var serviceProvider = services.BuildServiceProvider();

        // Creates a scope to get the database context
        var databaseContextFactory = serviceProvider
                                        .GetRequiredService<IDbContextFactory<PsmContext>>();

        // Creates instance of the database context
        using PsmContext context = databaseContextFactory.CreateDbContext();
        context.Database.EnsureCreated();
        context.Database.Migrate();
    }

    public static void PushPsm(PsmContext context, PSM psm)
    {
        context.Add(psm);
        context.SaveChanges();
    }

    public static void PushPsmCollection(PsmContext context, IEnumerable<PSM> psmCollection)
    {
        context.AddRange(psmCollection);
        context.SaveChanges();
    }

    /// <summary>
    /// Purpose of this method is to prepare a collection of PSMs that are candidates for further processing.
    /// If the full sequence is already in the database, then the PSM will be pushed without further processing.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="baseSequence"></param>
    /// <returns></returns>
    public static IEnumerable<PSM> PrepareCandidates(PsmContext context, IEnumerable<PSM> psmCollection)
    {
        List<PSM> newPsms = new List<PSM>();
        List<PSM> existingPsms = new List<PSM>();

        foreach (PSM psm in psmCollection)
        {
            var existingPsm = context.PSMs.FirstOrDefault(p => p.FullSequence == psm.FullSequence);

            if (existingPsm == null)
            {
                newPsms.Add(psm);
            }
            else
            {
                // Take the median of the two PSMs retention times

                existingPsm.ScanRetentionTime = Statistics.Median(new double[]
                {
                    existingPsm.ScanRetentionTime, psm.ScanRetentionTime
                });

                // Take the median integer of both PSMs scan numbers
                existingPsm.PrecursorScanNumber = (int)Statistics.Median(new double[]
                {
                    existingPsm.PrecursorScanNumber, psm.PrecursorScanNumber
                });

                // Remove the existing PSM from the database
                context.PSMs.Remove(existingPsm);

                existingPsms.Add(existingPsm);
            }
        }

        // Merge the new and existing Psms
        newPsms.AddRange(existingPsms);

        return newPsms;
    }

    public static void PrepareCandidatesAndCommit(PsmContext context, List<PSM> psmCollection)
    {
        IEnumerable<PSM> newPsms = PrepareCandidates(context, psmCollection);
        PushPsmCollection(context, newPsms);
    }

    public static IEnumerable<PSM> FetchBaseSequences(PsmContext context, string baseSequence)
    {
        return context.PSMs.Where(psm => psm.BaseSequence == baseSequence);
    }

    public static IEnumerable<PSM> FetchFullSequences(PsmContext context, string fullSequence)
    {
        return context.PSMs.Where(psm => psm.FullSequence == fullSequence);
    }

    /// <summary>
    /// TODO: This method will contain the logic to remove Psms from the collection that do not fit the criteria.
    /// </summary>
    /// <param name="psmCollection"></param>
    /// <returns></returns>
    private static List<PSM> CheckIn(IEnumerable<PSM> psmCollection)
    {
        // Get rid of ambiguous Psms
        psmCollection = psmCollection
            .Where(psm => psm.AmbiguityLevel == "1");

        return psmCollection.ToList();
    }
}
