using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Database.Migrations
{
    public partial class InitialCreate : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "PSMs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    FileName = table.Column<string>(type: "TEXT", nullable: false),
                    PrecursorScanNumber = table.Column<int>(type: "INTEGER", nullable: false),
                    ScanRetentionTime = table.Column<double>(type: "REAL", nullable: false),
                    TotalIonCurrent = table.Column<double>(type: "REAL", nullable: false),
                    PrecursorCharge = table.Column<int>(type: "INTEGER", nullable: false),
                    Notch = table.Column<int>(type: "INTEGER", nullable: false),
                    PrecursorMZ = table.Column<double>(type: "REAL", nullable: false),
                    PrecursorMass = table.Column<double>(type: "REAL", nullable: false),
                    Score = table.Column<double>(type: "REAL", nullable: false),
                    BaseSequence = table.Column<string>(type: "TEXT", nullable: false),
                    FullSequence = table.Column<string>(type: "TEXT", nullable: false),
                    AmbiguityLevel = table.Column<string>(type: "TEXT", nullable: false),
                    PeptideMonoisotopicMass = table.Column<string>(type: "TEXT", nullable: false),
                    ProteinAccession = table.Column<string>(type: "TEXT", nullable: false),
                    ProteinName = table.Column<string>(type: "TEXT", nullable: false),
                    GeneName = table.Column<string>(type: "TEXT", nullable: false),
                    OrganismName = table.Column<string>(type: "TEXT", nullable: false),
                    StartAndEndResidueInProtein = table.Column<string>(type: "TEXT", nullable: false),
                    MassErrorDaltons = table.Column<double>(type: "REAL", nullable: false),
                    PEP = table.Column<double>(type: "REAL", nullable: false),
                    QValue = table.Column<double>(type: "REAL", nullable: false),
                    PEPQvalue = table.Column<double>(type: "REAL", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_PSMs", x => x.Id);
                });
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "PSMs");
        }
    }
}
