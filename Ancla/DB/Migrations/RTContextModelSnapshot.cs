﻿// <auto-generated />
using DB;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace DB.Migrations
{
    [DbContext(typeof(RTContext))]
    partial class RTContextModelSnapshot : ModelSnapshot
    {
        protected override void BuildModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "6.0.29")
                .HasAnnotation("Relational:MaxIdentifierLength", 64);

            modelBuilder.Entity("DB.PSM", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    b.Property<string>("AmbiguityLevel")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("BaseSequence")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("FileName")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("FullSequence")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("GeneName")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("OrganismName")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<double>("PEP")
                        .HasColumnType("double");

                    b.Property<double>("PEPQvalue")
                        .HasColumnType("double");

                    b.Property<string>("PeptideMonoisotopicMass")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<int>("PrecursorCharge")
                        .HasColumnType("int");

                    b.Property<double>("PrecursorMZ")
                        .HasColumnType("double");

                    b.Property<double>("PrecursorMass")
                        .HasColumnType("double");

                    b.Property<string>("ProteinAccession")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<string>("ProteinName")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<double>("QValue")
                        .HasColumnType("double");

                    b.Property<int>("ScanNumber")
                        .HasColumnType("int");

                    b.Property<double>("ScanRetentionTime")
                        .HasColumnType("double");

                    b.Property<double>("Score")
                        .HasColumnType("double");

                    b.Property<string>("StartAndEndResidueInProtein")
                        .IsRequired()
                        .HasColumnType("longtext");

                    b.Property<double>("TotalIonCurrent")
                        .HasColumnType("double");

                    b.HasKey("Id");

                    b.ToTable("PSMs");
                });
#pragma warning restore 612, 618
        }
    }
}
