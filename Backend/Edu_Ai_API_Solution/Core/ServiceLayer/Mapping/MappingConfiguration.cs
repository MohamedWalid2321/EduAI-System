namespace ServiceLayer.Mapping
{
	public class MappingConfiguration : IRegister
	{
		public void Register(TypeAdapterConfig config)
		{
			config.NewConfig<Course, CourseResponseDto>()
				.Map(dest => dest.semster, src => src.semster.ToString());
			// Configure CourseRequestDto to Course mapping with enum conversion
			config.NewConfig<CourseRequestDto, Course>()
				.Map(dest => dest.semster, src => src.semster)
				.Map(dest => dest.IsPublished, src => true)
				.Map(dest => dest.AcademicLevel, src => src.academicLevel);

			TypeAdapterConfig<Course, FullCourseResponse>
			.NewConfig()
			.Map(dest => dest.Assesment, src => src.Assessments);
			// Configure AssesmentDto to Assessment mapping with enum conversion
			config.NewConfig<AssesmentDto, Assessment>()
				.Map(dest => dest.AssType, src => src.AssType);
			// Mapping Email To UserName in ApplicationUser
			config.NewConfig<RegisterRequest, ApplicationUser>()
			.Map(dest => dest.UserName, src => src.Email);
			config.NewConfig<ApplicationUser, UserProfileResponse>()
				.Map(dest => dest.AcademicYear, src => src.AcademicYear.ToString());
			config.NewConfig<RoleRequest, ApplicationRole>()
				.Map(dest => dest.IsEnrollable, src => src.IsEnrollable);

			config.NewConfig<(ApplicationUser user, IList<string> roles), UserResponse>()
			.Map(dest => dest, src => src.user)
			.Map(dest => dest.Roles, src => src.roles)
			.Map(dest => dest.AcademicYear, src => src.user.AcademicYear.HasValue ? src.user.AcademicYear.Value.ToString() : null)
			.Map(dest => dest.DepartmentId, src => src.user.DepartmentId ?? 0);

			config.NewConfig<CreateUserRequest, ApplicationUser>()
			.Map(dest => dest.UserName, src => src.Email)
			.Map(dest => dest.EmailConfirmed, src => true);

			config.NewConfig<UpdateUserRequest, ApplicationUser>()
				.Map(dest => dest.UserName, src => src.Email)
				.Map(dest => dest.NormalizedUserName, src => src.Email.ToUpper());

			config.NewConfig<QuestionRequestDto, QuizQuestion>()
				.Map(dest => dest.QuestionChoices, src => src.QuestionChoices
					.Select((answer, index) => new QuestionChoices
					{
						ChoiceText = answer,
						IsCorrect = index == src.CorrectAnswerIndex
					}).ToList());
			TypeAdapterConfig<ApplicationUser, InstructorsDetailsResponse>
				.NewConfig()
				.Map(dest => dest.FullName, src => $"Dr. {src.FirstName} {src.LastName}");

			// Lecture mappings
			config.NewConfig<Lecture, LectureResponse>()
				.Map(dest => dest.CreatedByName, src => $"{src.CreatedBy.FirstName} {src.CreatedBy.LastName}");
		}
	}
}
