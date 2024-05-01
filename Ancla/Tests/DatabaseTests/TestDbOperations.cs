using AnchorCommandLine;
using Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

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

        var builder = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false);

        IConfiguration config = builder.Build();

        var connectionString = config.GetSection("DatabaseConnection")
            .Get<DatabaseConnection>();

        var optionsBuilder = new DbContextOptionsBuilder<PsmContext>();
        optionsBuilder.UseMySQL(connectionString.ConnectionString);

        using (var context = new PsmContext(optionsBuilder.Options))
        {
            DbOperations.AddPsm(psm[0], context);
        }

    }
}

