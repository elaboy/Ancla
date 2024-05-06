﻿// <auto-generated />
using Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace Database.Migrations
{
    [DbContext(typeof(PsmContext))]
    [Migration("20240506171237_initForAllPsms2")]
    partial class initForAllPsms2
    {
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder.HasAnnotation("ProductVersion", "6.0.29");

            modelBuilder.Entity("AnchorLib.PSM", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<string>("AmbiguityLevel")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("BaseSequence")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("FileName")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("FullSequence")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("GeneName")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("MassErrorDaltons")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("Notch")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("OrganismName")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<double>("PEP")
                        .HasColumnType("REAL");

                    b.Property<double>("PEPQvalue")
                        .HasColumnType("REAL");

                    b.Property<string>("PeptideMonoisotopicMass")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<int>("PrecursorCharge")
                        .HasColumnType("INTEGER");

                    b.Property<double>("PrecursorMZ")
                        .HasColumnType("REAL");

                    b.Property<double>("PrecursorMass")
                        .HasColumnType("REAL");

                    b.Property<int>("PrecursorScanNumber")
                        .HasColumnType("INTEGER");

                    b.Property<string>("ProteinAccession")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<string>("ProteinName")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<double>("QValue")
                        .HasColumnType("REAL");

                    b.Property<double>("ScanRetentionTime")
                        .HasColumnType("REAL");

                    b.Property<double>("Score")
                        .HasColumnType("REAL");

                    b.Property<string>("StartAndEndResidueInProtein")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<double>("TotalIonCurrent")
                        .HasColumnType("REAL");

                    b.HasKey("Id");

                    b.ToTable("PSMs");
                });
#pragma warning restore 612, 618
        }
    }
}
