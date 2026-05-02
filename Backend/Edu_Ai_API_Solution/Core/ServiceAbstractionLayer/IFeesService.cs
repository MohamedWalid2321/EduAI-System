using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace ServiceAbstractionLayer
{
    public interface IFeesService
    {
        Task<FeeResponseDto> SetFeesAsync(FeeRequestDto newFee, CancellationToken cancellationToken = default);
        Task<FeeResponseDto> UpdateFeesAsync(int feeId, FeeRequestDto newFee, CancellationToken cancellationToken = default);
        Task<IEnumerable<FeeResponseDto>> GetByAcademicYearAsync(int academicYearId, CancellationToken cancellationToken = default);
    }
}
