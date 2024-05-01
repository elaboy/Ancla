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
}

