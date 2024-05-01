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
}



