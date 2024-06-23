using Ancla.Models;
using Microsoft.EntityFrameworkCore;

namespace Ancla;
public class DbContext : Microsoft.EntityFrameworkCore.DbContext
{
    public DbSet<PSM> PSMs { get; set; }

    public DbContext()
    {
    }

    public DbContext(DbContextOptions<DbContext> options)
        : base(options)
    {
    }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlite("Data Source = D:anchor.db");
    }
}
