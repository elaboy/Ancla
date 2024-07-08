using System.Data;
using System.Globalization;
using AnchorCommandLine;
using CsvHelper;

namespace TestCMD
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TestA549CalibrationConstructor()
        {
            var calibrator = new Calibrator(@"D:\MannPeptideResults\A549_AllPSMs.psmtsv");
            Assert.AreEqual(0, calibrator.LibraryRetentionTimes.Count);
        }

        [Test]
        public void TestA549CalibrationCalibration()
        {
            var calibrator = new Calibrator(@"D:\MannPeptideResults\A549_AllPSMs.psmtsv");
            calibrator.Calibrate();

            var calibrations = calibrator.FileLoggers[0].FileWiseCalibrations;

            // save these as a table to export as  csv 
            var table = new DataTable();
            table.Columns.Add("Raw File");
            table.Columns.Add("Sequence");
            table.Columns.Add("Retention Time");

            foreach (var rawFile in calibrations)
            {
                foreach (var sequence in rawFile.Value)
                {
                    table.Rows.Add(rawFile.Key, sequence.Item1, sequence.Item2);
                }
            }
             
            //save DataTable as csv 
            using var writer = new StreamWriter(@"D:\MannPeptideResults\A549_Calibrations.csv");
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                foreach(DataColumn column in table.Columns)
                {
                    csv.WriteField(column.ColumnName);
                }
                csv.NextRecord();
                foreach (DataRow row in table.Rows)
                {
                    for (var i = 0; i < table.Columns.Count; i++)
                    {
                        csv.WriteField(row[i]);
                    }
                    csv.NextRecord();
                }
            }
            
            Assert.AreEqual(0, calibrator.LibraryRetentionTimes.Count);
        }
    }
}