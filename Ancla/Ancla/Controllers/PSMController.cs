using Ancla.Models;
using Database;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Ancla.Controllers
{
    public class PSMController : Controller
    {
        private readonly PsmContext _context;

        public PSMController(PsmContext context)
        {
            _context = context;
        }

        // GET: PSM
        public async Task<IActionResult> Index()
        {
            return _context.PSMs != null ?
                        View(await _context.PSMs.ToListAsync()) :
                        Problem("Entity set 'RTContext.PSMs'  is null.");
        }

        // GET: PSM/Details/5
        public async Task<IActionResult> Details(int? id)
        {
            if (id == null || _context.PSMs == null)
            {
                return NotFound();
            }

            var pSM = await _context.PSMs
                .FirstOrDefaultAsync(m => m.Id == id);
            if (pSM == null)
            {
                return NotFound();
            }

            return View(pSM);
        }

        // GET: PSM/Create
        public IActionResult AddPsm()
        {
            return View(new PSM());
        }

        // POST: PSM/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        // Takes the uploaded file and reads it into the database
        public async Task<IActionResult> AddPsm(IFormFile file)
        {
            if (ModelState.IsValid)
            {
                //use the file path to read the file
                var psms = new Readers.PsmFromTsvFile(file.FileName);
                psms.LoadResults();
                foreach (var psm in psms)
                {
                    _context.Add(new PSM()
                    {
                        AmbiguityLevel = psm.AmbiguityLevel,
                        BaseSequence = psm.BaseSeq,
                        FileName = psm.FileNameWithoutExtension,
                        FullSequence = psm.FullSequence,
                        GeneName = psm.GeneName,
                        OrganismName = psm.OrganismName,
                        PeptideMonoisotopicMass = psm.PeptideMonoMass,
                        PrecursorCharge = psm.PrecursorCharge,
                        PrecursorMass = psm.PrecursorMass,
                        PrecursorMZ = psm.PrecursorMz,
                        ProteinAccession = psm.ProteinAccession,
                        ProteinName = psm.ProteinName,
                        QValue = psm.QValue,
                        Score = psm.Score,
                        ScanNumber = psm.PrecursorScanNum,
                        ScanRetentionTime = psm.RetentionTime.Value,
                        StartAndEndResidueInProtein = psm.StartAndEndResiduesInProtein,
                        TotalIonCurrent = psm.TotalIonCurrent.Value,
                        PEP = psm.PEP,
                        PEPQvalue = psm.PEP_QValue
                    });
                }
                //_context.Add(new PSM());
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }
            return View();
        }

        // GET: PSM/Edit/5
        public async Task<IActionResult> Edit(int? id)
        {
            if (id == null || _context.PSMs == null)
            {
                return NotFound();
            }

            var pSM = await _context.PSMs.FindAsync(id);
            if (pSM == null)
            {
                return NotFound();
            }
            return View(pSM);
        }

        // POST: PSM/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, [Bind("Id,FileName,ScanNumber,ScanRetentionTime,TotalIonCurrent,PrecursorCharge,PrecursorMZ,PrecursorMass,Score,BaseSequence,FullSequence,AmbiguityLevel,PeptideMonoisotopicMass,ProteinAccession,ProteinName,GeneName,OrganismName,StartAndEndResidueInProtein,PEP,QValue,PEPQvalue")] PSM pSM)
        {
            if (id != pSM.Id)
            {
                return NotFound();
            }

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Update(pSM);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!PSMExists(pSM.Id))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }
            return View(pSM);
        }

        // GET: PSM/Delete/5
        public async Task<IActionResult> Delete(int? id)
        {
            if (id == null || _context.PSMs == null)
            {
                return NotFound();
            }

            var pSM = await _context.PSMs
                .FirstOrDefaultAsync(m => m.Id == id);
            if (pSM == null)
            {
                return NotFound();
            }

            return View(pSM);
        }

        // POST: PSM/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            if (_context.PSMs == null)
            {
                return Problem("Entity set 'RTContext.PSMs'  is null.");
            }
            var pSM = await _context.PSMs.FindAsync(id);
            if (pSM != null)
            {
                _context.PSMs.Remove(pSM);
            }

            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        private bool PSMExists(int id)
        {
            return (_context.PSMs?.Any(e => e.Id == id)).GetValueOrDefault();
        }
    }
}
