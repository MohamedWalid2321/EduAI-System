using ServiceLayer.Specifications.AcademicYearSpecifications;
using Shared.Dtos.AcademicYearDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class AcademicYearService(IUnitOfWork unitOfWork) : IAcademicYearService
    {
        private readonly IUnitOfWork _unitOfWork = unitOfWork;

        public async Task<AcademicYearDto> CreateAsync(string name, CancellationToken cancellationToken = default)
        {
            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();

            if (string.IsNullOrWhiteSpace(name))
                throw new Exception("Academic year name is required");

            var entity = new AcademicYear
            {
                Name = name
            };

            await  academicYearRepo.AddAsync(entity, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return entity.Adapt< AcademicYearDto>();
        }

        public async Task<List<AcademicYearDto>> GetAllAsync(CancellationToken cancellationToken = default)
        {
            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();

            var academicYearSpecification = new AcademicYearSpecifications();
            var years = await academicYearRepo.GetAllAsync(academicYearSpecification, cancellationToken);

            return years.Adapt<List<AcademicYearDto>>();
        }

        public async Task<AcademicYearDto> GetByIdAsync(int id, CancellationToken cancellationToken = default)
        {
            var academicYearRepo = _unitOfWork.GetRepository<AcademicYear, int>();
            var academicYearSpecification = new AcademicYearSpecifications(id);
            var entity = await academicYearRepo.GetByIdAsync(academicYearSpecification, cancellationToken);

            if (entity == null)
                throw new Exception("Academic year not found");

            return entity.Adapt<AcademicYearDto>();
        }
    }
}
