using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class AcademicYearController(IServiceManager serviceManager, ICacheService cacheService) : ApiControllerBase
    {
        private const string AcademicYearsPattern = "/api/academicyear*";

        [HttpGet]
        [Cache(3600)]
        public async Task<IActionResult> GetAll(CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.GetAllAsync(cancellationToken);
            return Ok(result);
        }

        [HttpGet("{id}")]
        [Cache(3600)]
        public async Task<IActionResult> GetById(int id, CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.GetByIdAsync(id, cancellationToken);
            if (result == null)
                return NotFound("Academic year not found");
            return Ok(result);
        }

        [HttpPost]
        public async Task<IActionResult> Create([FromBody] string name, CancellationToken cancellationToken)
        {
            var result = await serviceManager.AcademicYearService.CreateAsync(name, cancellationToken);
            await cacheService.RemoveByPatternAsync(AcademicYearsPattern);
            return Ok(result);
        }
    }
}
