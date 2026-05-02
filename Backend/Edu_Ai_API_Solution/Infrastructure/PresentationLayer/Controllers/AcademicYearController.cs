using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class AcademicYearController(IServiceManager serviceManager) : ApiControllerBase
    {
        [HttpPost]
        public async Task<IActionResult> Create([FromBody] string name, CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.CreateAsync(name, cancellationToken);
            return Ok(result);
        }

        [HttpGet]
        public async Task<IActionResult> GetAll(CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.GetAllAsync(cancellationToken);
            return Ok(result);
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetById(int id, CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.GetByIdAsync(id, cancellationToken);

            if (result == null)
                return NotFound("Academic year not found");

            return Ok(result);
        }

    }
}
