
using Shared.Dtos.AcademicYearDto;
using Shared.Dtos.FeeDto;

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
				.Map(dest => dest.IsPublished, src => true);
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
				.Map(dest => dest.AcademicYear, src => src.AcademicYearEnum.ToString());

			config.NewConfig<(ApplicationUser user, IList<string> roles), UserResponse>()
			.Map(dest => dest, src => src.user)
			.Map(dest => dest.Roles, src => src.roles)
            .Map(dest => dest.AcademicYear,
					src => src.user.AcademicYearEnum.HasValue
							? src.user.AcademicYearEnum.Value.ToString()
							: "Not Assigned")
            .Map(dest => dest.DepartmentId , src => src.user.DepartmentId ?? 0);

			config.NewConfig<CreateUserRequest, ApplicationUser>()
			.Map(dest => dest.UserName, src => src.Email)
			.Map(dest => dest.EmailConfirmed, src => true)
			;

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

            config.NewConfig<Fee, FeeResponseDto>()
				.Map(dest => dest.academicYearId, src => src.AcademicYearId)
				.Map(dest => dest.amount, src => src.Amount)
				.Map(dest => dest.name, src => src.Name.ToString());

            config.NewConfig<FeeRequestDto, Fee>()
			      .Map(dest => dest.Name, src => Enum.Parse<FeeType>(src.name, true));

            config.NewConfig<AcademicYear, AcademicYearDto>()
    .Map(dest => dest.Id, src => src.Id)
    .Map(dest => dest.Name, src => src.Name)
    .Map(dest => dest.fees, src => src.Fees);

        }
	}
}
