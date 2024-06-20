using System.Runtime.InteropServices.ComTypes;
using AnchorLib;
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
        using var context = databaseContextFactory.CreateDbContext();
        context.Database.EnsureCreated();
        context.Database.Migrate();
    }

    public static void PushPsm(PsmContext context, PSM psm)
    {
        context.Add(psm);
        context.SaveChanges();
    }

    public static void PushPsmCollection(PsmContext context, List<PSM> psmCollection)
    {
        context.AddRange(psmCollection);
        context.SaveChanges();
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
