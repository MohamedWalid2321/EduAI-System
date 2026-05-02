using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class FeeController(IServiceManager serviceManager) : ApiControllerBase
    {
        [HttpPost]
        public async Task<IActionResult> CreateFee([FromForm] FeeRequestDto dto, CancellationToken cancellationToken)
        {
            var result = await serviceManager.FeesService.SetFeesAsync(dto, cancellationToken);
            return Ok(result);
        }

        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateFee(int id, [FromForm] FeeRequestDto dto, CancellationToken cancellationToken)
        {
            var result = await serviceManager.FeesService.UpdateFeesAsync(id, dto, cancellationToken);
            return Ok(result);
        }

        [HttpGet("academic-year/{academicYearId}")]
        public async Task<IActionResult> GetByAcademicYear(int academicYearId, CancellationToken cancellationToken)
        {
            var result = await serviceManager.FeesService.GetByAcademicYearAsync(academicYearId, cancellationToken);
            return Ok(result);
        }
    }
}
