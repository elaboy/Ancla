﻿using AnchorLib;
using ThermoFisher.CommonCore.Data;

namespace Database;

public static class DbOperations
{
    public static string ConnectionString = @"Data Source = D:\anchor_AllPsms.db";
    public static void AddPsms(PsmContext context, List<PSM> psms)
    {
        foreach (var psm in psms)
        {
            context.Add(psm);
        }
        context.SaveChanges();
    }

    public static void AddPsm(PsmContext context, PSM psm)
    {
        context.Add(psm);
        context.SaveChanges();
    }

    public static void AddPsmsNonRedundant(PsmContext context, List<PSM> psms)
    {
        //one bulk transaction instead of multiple transactions (per psm) 
        var psmsInDB = context.PSMs.ToList();

        List<PSM> psmsToUpload = new List<PSM>();

        // empty database, dont check for redundancy, else upload every psm with qvalue <= 0.01
        if (psmsInDB.IsNullOrEmpty())
        {
            psmsToUpload.AddRange(psms.Where(p => p.QValue <= 0.01));
        }
        else
        {

            Parallel.ForEach(psms, psm =>
            {
                if (psm.QValue <= 0.01)
                {
                    // Fetch existing data in bulk
                    var existingData = psmsInDB
                        .FirstOrDefault(p => p.FileName == psm.FileName &&
                                             p.FullSequence == psm.FullSequence);
                    //.ToList();

                    if (existingData == null)
                    {
                        // Add new PSM if not found in the database
                        psmsToUpload.Add(psm);
                    }
                }

            });
        }

        // transaction to add all new psms to the database
        using (var transaction = context.Database.BeginTransaction())
        {

            // Add all new PSMs to the database
            context.AddRange(psmsToUpload);
            context.SaveChanges();
            transaction.Commit();
        }
    }

    public static List<PSM> FetchAnchors(PsmContext context, bool orderedByElution = false)
    {
        if (orderedByElution)
        {
            var anchors = context.PSMs.ToList();

            // group anchors by full sequence and average the retention times
            var groupedAnchors = anchors.GroupBy(a => a.FullSequence)
                .Select(g => new PSM
                {
                    FullSequence = g.Key,
                    ScanRetentionTime = g.Average(a => a.ScanRetentionTime)
                }).ToList();

            return groupedAnchors
                .OrderBy(p => p.ScanRetentionTime)
                .ToList();
        }
        else
        {
            var anchors = context.PSMs.ToList();

            // group anchors by full sequence and average the retention times
            var groupedAnchors = anchors.GroupBy(a => a.FullSequence)
                .Select(g => new PSM
                {
                    FullSequence = g.Key,
                    ScanRetentionTime = g.Average(a => a.ScanRetentionTime)
                }).ToList();

            return groupedAnchors;
        }
    }
}

public class ArchorOperations
{
    public void IdentifyAnchors()
    {
    }


}


