using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class FeeController(IServiceManager serviceManager) :ApiControllerBase
    {
        

        [HttpPost]
        public async Task<IActionResult> CreateFee([FromForm] FeeRequestDto dto)
        {
            var result = await serviceManager.FeesService.SetFeesAsync(dto);
            return Ok(result);
        }

        [HttpPut("{id}")]
        public async Task<IActionResult> UpdateFee(int id, [FromForm] FeeRequestDto dto)
        {
            var result = await serviceManager.FeesService.UpdateFeesAsync(id, dto);
            return Ok(result);
        }

        [HttpGet("academic-year/{academicYearId}")]
        public async Task<IActionResult> GetByAcademicYear(int academicYearId)
        {
            var result = await serviceManager.FeesService.GetByAcademicYearAsync(academicYearId);
            return Ok(result);
        }



    }
}
