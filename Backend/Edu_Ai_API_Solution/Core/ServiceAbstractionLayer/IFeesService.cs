using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IFeesService
    {
        Task<FeeResponseDto> SetFeesAsync(FeeRequestDto newFee);
        Task<FeeResponseDto> UpdateFeesAsync(int feeId , FeeRequestDto newFee);
        Task<IEnumerable<FeeResponseDto>> GetByAcademicYearAsync(int academicYearId);
        
    }
}
