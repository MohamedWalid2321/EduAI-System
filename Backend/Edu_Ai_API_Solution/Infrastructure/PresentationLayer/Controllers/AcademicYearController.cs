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
        public async Task<IActionResult> Create([FromBody] string name)
        {
            var result = await serviceManager.AcademicYearService.CreateAsync(name);
            return Ok(result);
        }

        [HttpGet]
        public async Task<IActionResult> GetAll()
        {
            var result = await serviceManager.AcademicYearService.GetAllAsync();
            return Ok(result);
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetById(int id)
        {
            var result = await serviceManager.AcademicYearService.GetByIdAsync(id);

            if (result == null)
                return NotFound("Academic year not found");

            return Ok(result);
        }

    }
}
