using AnchorLib;

namespace Database;

public static class DbOperations
{
    public static void AddPsms(List<PSM> psms, PsmContext context)
    {
        foreach (var psm in psms)
        {
            context.Add(psm);
        }
        context.SaveChanges();
    }
}



