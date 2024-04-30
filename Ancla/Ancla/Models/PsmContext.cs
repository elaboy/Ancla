using Microsoft.EntityFrameworkCore;

namespace Ancla.Models
{
    public class PsmContext : DbContext
    {
        public PsmContext(DbContextOptions<PsmContext> options) 
            : base(options)
        {

        }

        public DbSet<PSM> PSMs { get; set; }
    }
}