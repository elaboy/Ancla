using AnchorLib;
using Microsoft.EntityFrameworkCore;

namespace Database;

public class PsmContext : DbContext
{
    public PsmContext(DbContextOptions<PsmContext> options)
        : base(options)
    {

    }

    public DbSet<PSM> PSMs { get; set; }
}