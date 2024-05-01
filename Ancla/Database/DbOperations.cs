using AnchorLib;

namespace Database;

public static class DbOperations
{
    //public static string ConnectionString = 
    public static void AddPsms(List<PSM> psms, PsmContext context)
    {
        foreach (var psm in psms)
        {
            context.Add(psm);
        }
        context.SaveChanges();
    }

    public static void AddPsm(PSM psm, PsmContext context)
    {
        context.Add(psm);
        context.SaveChanges();
    }
}



