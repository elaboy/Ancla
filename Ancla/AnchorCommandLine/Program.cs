using AnchorLib;
using Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

namespace AnchorCommandLine;

public class Program
{
    public static void Main(string[] args)
    {
        List<string> paths = new List<string>();

        foreach (var arg in args)
        {
            paths.Add(arg);
        }

        // get the configuration from the appsettings.json file
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
            var psmService = new PsmService(context);
            var psms = psmService.GetPsms(paths);

        }
    }
}

public class DatabaseConnection
{
    public string ConnectionString { get; set; }
}