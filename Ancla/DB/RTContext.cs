using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using MySql.Data.MySqlClient;

namespace DB
{
    public class RTContext : DbContext
    {
        public RTContext(DbContextOptions<RTContext> options) 
            : base(options)
        {

        }

        public DbSet<PSM> PSMs { get; set; }
    }
}