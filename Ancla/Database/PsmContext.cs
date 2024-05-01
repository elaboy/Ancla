using AnchorLib;
using Microsoft.EntityFrameworkCore;

namespace Database;

public class PsmContext : DbContext
{
    public PsmContext(DbContextOptions<PsmContext> options)
        : base(options)
    {

    }

    protected override void OnConfiguring(DbContextOptionsBuilder options)

        => options.UseSqlite();

    public DbSet<PSM> PSMs { get; set; }
}