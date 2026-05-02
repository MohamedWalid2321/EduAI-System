using DomainLayer.Models;
using System.Net.Mail;

namespace ServiceLayer.Services
{
	public class ContentService(IUnitOfWork unitOfWork , IFileStorageService fileStorageService) : IContentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		

		public async Task<ContentResponseDto> AddContentForCourse(int courseId, ContentRequestDto contentRequest, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var course = await courseRepository.GetByIdAsync(courseId, cancellationToken);
			if (course is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			var contentEntity = contentRequest.Adapt<Content>();
			contentEntity.CourseId = courseId;
			await contentRepository.AddAsync(contentEntity, cancellationToken);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			return contentEntity.Adapt<ContentResponseDto>();
		}

		public async Task UpdateContentForCourse(int contentId, ContentRequestDto contentRequest, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var ContentSpecifics = new ContentSpecification(contentId);
			var foundedContentEntity = await contentRepository.GetByIdAsync(ContentSpecifics, cancellationToken);
			if (foundedContentEntity is null)
			{
				throw new ContentNotFoundException(contentId);
			}
			foundedContentEntity.Title = contentRequest.Title;
			foundedContentEntity.Body = contentRequest.Body;
			contentRepository.Update(foundedContentEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}
		public async Task<IEnumerable<ContentResponseDto>> GetAllContentsByCourseIdAsync(int courseId, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var contentSpecification = new ContentByCourseIdSpecification(courseId);
			var ContentEntities = await contentRepository.GetAllAsync(contentSpecification, cancellationToken);
			if (ContentEntities is null || !ContentEntities.Any())
			{
				throw new ContentsInCourseNotFoundException(courseId);
			}
			return ContentEntities.Adapt<IEnumerable<ContentResponseDto>>();
		}

		public async Task<ContentResponseDto> GetContentByIdAsync(int contentId, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var contentSpec = new ContentSpecification(contentId);
			var contentEntity = await contentRepository.GetByIdAsync(contentSpec, cancellationToken);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(contentId);
			}
			return contentEntity.Adapt<ContentResponseDto>();
		}
		public async Task DeleteContentAsync(int contentId, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var ContentSpecifics = new ContentSpecification(contentId);
			var contentEntity = await contentRepository.GetByIdAsync(ContentSpecifics, cancellationToken);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(contentId);
			}
			if (contentEntity.ContentAttachments.Any()) {
				foreach (var attachment in contentEntity.ContentAttachments)
				{
					if (!string.IsNullOrEmpty(attachment.FileUrl)) {
						await _fileStorageService.DeleteFileAsync(attachment.FileUrl);
					}
				}
			}
			contentRepository.Delete(contentEntity!);
			await _unitOfWork.SaveChangesAsync(cancellationToken);	
		}

		

		public async Task RemoveAttachment(Guid AttachmentId, CancellationToken cancellationToken = default)
		{
			var attachmentRepository = _unitOfWork.GetRepository<ContentAttachment, Guid>();
			 var attachmentEntity = await attachmentRepository.GetByIdAsync(AttachmentId, cancellationToken);
			if (attachmentEntity is null)
			{
				throw new ContentAttachmentNotFoundException(AttachmentId);
			}
			if (!string.IsNullOrEmpty(attachmentEntity.FileUrl))
			{
				await _fileStorageService.DeleteFileAsync(attachmentEntity.FileUrl);
			}
			attachmentRepository.Delete(attachmentEntity!);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}
		public async Task<ContentResponseDto> AddAttachmentToContent(int ContentId, List<IFormFile?> Files, CancellationToken cancellationToken = default)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var attachmentRepository = _unitOfWork.GetRepository<ContentAttachment, Guid>();
			var contentSpec = new ContentSpecification(ContentId);
			var contentEntity = await contentRepository.GetByIdAsync(contentSpec, cancellationToken);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(ContentId);
			}
			foreach (var file in Files)
			{
				if (file is not null && file.Length > 0)
				{
					using var stream = file.OpenReadStream();
					var fileUrl = await _fileStorageService.UploadFileAsync(
						stream,
						file.FileName,
						$"Courses/{contentEntity.Course.Title}/Contents/{ContentId}",
						file.ContentType
					);
					var attachment = new ContentAttachment
					{
						FileName = file.FileName,
						FileUrl = fileUrl,
						ContentType = file.ContentType,
						ContentId = ContentId
					};
					await attachmentRepository.AddAsync(attachment, cancellationToken);
				}
			}
			await _unitOfWork.SaveChangesAsync(cancellationToken);
			// Reload with attachments
			var updatedContent = await contentRepository.GetByIdAsync(contentSpec, cancellationToken);
			return updatedContent!.Adapt<ContentResponseDto>();
		}

	}
}
