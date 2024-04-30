using Microsoft.EntityFrameworkCore.Migrations;
using MySql.EntityFrameworkCore.Metadata;

#nullable disable

namespace DB.Migrations
{
    public partial class InitDB : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterDatabase()
                .Annotation("MySQL:Charset", "utf8mb4");

            migrationBuilder.CreateTable(
                name: "PSMs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("MySQL:ValueGenerationStrategy", MySQLValueGenerationStrategy.IdentityColumn),
                    FileName = table.Column<string>(type: "longtext", nullable: false),
                    ScanNumber = table.Column<int>(type: "int", nullable: false),
                    ScanRetentionTime = table.Column<double>(type: "double", nullable: false),
                    TotalIonCurrent = table.Column<double>(type: "double", nullable: false),
                    PrecursorCharge = table.Column<int>(type: "int", nullable: false),
                    PrecursorMZ = table.Column<double>(type: "double", nullable: false),
                    PrecursorMass = table.Column<double>(type: "double", nullable: false),
                    Score = table.Column<double>(type: "double", nullable: false),
                    BaseSequence = table.Column<string>(type: "longtext", nullable: false),
                    FullSequence = table.Column<string>(type: "longtext", nullable: false),
                    AmbiguityLevel = table.Column<string>(type: "longtext", nullable: false),
                    PeptideMonoisotopicMass = table.Column<string>(type: "longtext", nullable: false),
                    ProteinAccession = table.Column<string>(type: "longtext", nullable: false),
                    ProteinName = table.Column<string>(type: "longtext", nullable: false),
                    GeneName = table.Column<string>(type: "longtext", nullable: false),
                    OrganismName = table.Column<string>(type: "longtext", nullable: false),
                    StartAndEndResidueInProtein = table.Column<string>(type: "longtext", nullable: false),
                    PEP = table.Column<double>(type: "double", nullable: false),
                    QValue = table.Column<double>(type: "double", nullable: false),
                    PEPQvalue = table.Column<double>(type: "double", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_PSMs", x => x.Id);
                })
                .Annotation("MySQL:Charset", "utf8mb4");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "PSMs");
        }
    }
}
