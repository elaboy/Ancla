using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Database.Migrations
{
    public partial class initForAllPsms : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateIndex(
                name: "IX_PSMs_FileName_FullSequence",
                table: "PSMs",
                columns: new[] { "FileName", "FullSequence" },
                unique: true);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_PSMs_FileName_FullSequence",
                table: "PSMs");
        }
    }
}
