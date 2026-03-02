


namespace ServiceLayer.Services
{
	public class CourseService(IUnitOfWork unitOfWork,IFileStorageService fileStorageService) : ICourseService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		public async Task<CourseResponseDto> CreateOrUpdateCourseAsync(CourseRequestDto CreatedcourseDto, IFormFile? ImageFile)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = CreatedcourseDto.Adapt<Course>();
			if (CreatedcourseDto.Id > 0)
			{
				//Update 
				var FoundedCourseEntity = await courseRepository.GetByIdAsync(CreatedcourseDto.Id);
				if (FoundedCourseEntity is null)
				{
					throw new CourseNotFoundException(CreatedcourseDto.Id);
				}
				if (ImageFile is not null && ImageFile.Length > 0)
				{
					using var stream = ImageFile.OpenReadStream();
					var imagePath = await _fileStorageService.UploadFileAsync(
						stream,
						ImageFile.FileName,
						"Courses/images",
						ImageFile.ContentType);
					courseEntity.ImageUrl = imagePath;
				}
				// Handle assessments for update
				if (CreatedcourseDto.Assesment != null && CreatedcourseDto.Assesment.Any())
				{
					var assessmentEntities = CreatedcourseDto.Assesment.Adapt<List<Assessment>>();
					foreach (var assessment in assessmentEntities)
					{
						assessment.CourseId = CreatedcourseDto.Id;
					}
					courseEntity.Assessments = assessmentEntities;
				}
				courseRepository.Update(courseEntity);
			}
			else {
				//create
				if (ImageFile is not null && ImageFile.Length > 0)
				{
					using var stream = ImageFile.OpenReadStream();
					var imagePath = await _fileStorageService.UploadFileAsync(
						stream,
						ImageFile.FileName,
						"Courses/images",
						ImageFile.ContentType);
					courseEntity.ImageUrl = imagePath;
				}
				if (CreatedcourseDto.Assesment != null && CreatedcourseDto.Assesment.Any())
				{
					var assessmentEntities = CreatedcourseDto.Assesment.Adapt<List<Assessment>>();
					courseEntity.Assessments = assessmentEntities;
				}
				await courseRepository.AddAsync(courseEntity);
			}
			await _unitOfWork.SaveChangesAsync();
			return courseEntity.Adapt<CourseResponseDto>();
		}
		public async Task DeleteCourseAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await courseRepository.GetByIdAsync(courseId);
			if (courseEntity is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			courseRepository.Delete(courseEntity!);
			await _unitOfWork.SaveChangesAsync();
		}
		public async Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync()
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecificatioin = new CourseSpecification();
			var courses =  await courseRepository.GetAllAsync(courseSpecificatioin);
			var courseDtos = courses.Adapt<IEnumerable<CourseResponseDto>>();
			return courseDtos;
		}
		public async Task<CourseResponseDto> GetCourseByIdAsync(int courseId)
		{
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecificatioin = new CourseSpecification(courseId);
			var course = await courseRepository.GetByIdAsync(courseSpecificatioin);
			if (course is null) 
			{
				throw new CourseNotFoundException(courseId);
			}
			var courseDto = course.Adapt<CourseResponseDto>();
			return courseDto;
		}
	}
}
