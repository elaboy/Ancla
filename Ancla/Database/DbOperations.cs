using AnchorLib;

namespace Database;

public static class DbOperations
{
    public static string ConnectionString = @"Data Source = D:\anchor.db";
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
    public static void AddPsmsNonRedundant(PsmContext context, List<PSM> psms)
    {
        //one bulk transaction instead of multiple transactions (per psm) 
        using (var transaction = context.Database.BeginTransaction())
        {
            foreach (var psm in psms)
            {
                // Fetch existing data in bulk
                var existingData = context.PSMs
                    .Where(p => p.FileName == psm.FileName &&
                                p.FullSequence == psm.FullSequence)
                    .ToList();

                if (existingData.Count == 0)
                {
                    // Add new PSM if not found in the database
                    context.Add(psm);
                }
            }
            context.SaveChanges();
            transaction.Commit();
        }
    }

    public static void FetchAnchorsFromDb()
    {

    }
}

public class ArchorOperations
{
    public void IdentifyAnchors()
    {
    }


}


