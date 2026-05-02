using DomainLayer.Models;
using Mapster;
using ServiceLayer.Specifications.FeeSpecifications;
using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class FeesService(IUnitOfWork unitOfWork) : IFeesService
    {
        private readonly IUnitOfWork _unitOfWork = unitOfWork;

        public async Task<FeeResponseDto> SetFeesAsync(FeeRequestDto newFee, CancellationToken cancellationToken = default)
        {
            if (newFee.Amount <= 0)
                throw new Exception("Invalid fee amount");

            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();
            var feeRepo = _unitOfWork.GetRepository<Fee, int>();

            var academicYear = await academicYearRepo.GetByIdAsync(newFee.AcademicYearId, cancellationToken);

            if (academicYear == null)
                throw new Exception("Academic year not found");

            var feeType = Enum.Parse<FeeType>(newFee.name);

            var feeSpecification = new FeeSpecifications(academicYear.Id, feeType);

            var existingFees = await feeRepo.GetAllAsync(feeSpecification, cancellationToken);

            if (existingFees.Any())
                throw new Exception("Fee already exists for this Academic Year");

            var feeEntity = newFee.Adapt<Fee>();

            feeEntity.AcademicYearId = academicYear.Id;

            await feeRepo.AddAsync(feeEntity, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return feeEntity.Adapt<FeeResponseDto>();
        }

        public async Task<FeeResponseDto> UpdateFeesAsync(int feeId, FeeRequestDto newFee, CancellationToken cancellationToken = default)
        {
            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();
            var feeRepo = _unitOfWork.GetRepository<Fee, int>();

            var feeEntity = await feeRepo.GetByIdAsync(feeId, cancellationToken);

            if (feeEntity == null)
                throw new Exception("Fee not found");

            feeEntity.Amount = newFee.Amount;
            feeEntity.FeeType = Enum.Parse<FeeType>(newFee.name, true);


            feeRepo.Update(feeEntity);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return feeEntity.Adapt<FeeResponseDto>();
        }

        public async Task<IEnumerable<FeeResponseDto>> GetByAcademicYearAsync(int academicYearId, CancellationToken cancellationToken = default)
        {
            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();
            var feeRepo = unitOfWork.GetRepository<Fee, int>();

            var feeSpecification = new FeeSpecifications(academicYearId);

            var feeEntities = await feeRepo.GetAllAsync(feeSpecification, cancellationToken);

            if (!feeEntities.Any())
                throw new Exception("No fees found");

            return feeEntities.Adapt<IEnumerable<FeeResponseDto>>();
        }


    }
}
