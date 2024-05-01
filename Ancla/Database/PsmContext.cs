using AnchorLib;
using Microsoft.EntityFrameworkCore;

namespace Database;

public class PsmContext : DbContext
{
    public PsmContext()
    {
        
    }

    public PsmContext(DbContextOptions<PsmContext> options)
        : base(options)
    {

    }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite(DbOperations.ConnectionString);

    public DbSet<PSM> PSMs { get; set; }
}