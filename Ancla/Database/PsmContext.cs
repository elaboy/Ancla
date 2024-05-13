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

    // protected override void OnConfiguring(DbContextOptionsBuilder options)
    //     => options.UseSqlite();

    //// Avoids duplications in the database todo: not working
    //protected override void OnModelCreating(ModelBuilder modelBuilder)
    //{
    //    modelBuilder.Entity<PSM>(psm =>
    //    {
    //        psm.HasIndex(p => new { p.FileName, p.FullSequence })
    //            .IsUnique();
    //    });
    //}

    public DbSet<PSM> PSMs { get; set; }
}