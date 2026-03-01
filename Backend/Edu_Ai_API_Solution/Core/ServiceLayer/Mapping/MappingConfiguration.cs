
namespace ServiceLayer.Mapping
{
	public class MappingConfiguration : IRegister
	{
		public void Register(TypeAdapterConfig config)
		{
			// Configure CourseDto to Course mapping with enum conversion
			config.NewConfig<CourseRequestDto, Course>()
				.Map(dest => dest.semster, src => Enum.Parse<Semster>(src.semster, true))
				.Map(dest => dest.CourseStatus, src => Enum.Parse<CourseStatus>(src.CourseStatus, true));

			// Configure Course to CourseDto mapping with enum to string conversion
			config.NewConfig<Course, CourseRequestDto>()
				.Map(dest => dest.semster, src => src.semster.ToString())
				.Map(dest => dest.CourseStatus, src => src.CourseStatus.ToString());

			// Configure CourseRequestDto to Course mapping with enum conversion
			config.NewConfig<CourseRequestDto, Course>()
				.Map(dest => dest.semster, src => Enum.Parse<Semster>(src.semster, true))
				.Map(dest => dest.CourseStatus, src => Enum.Parse<CourseStatus>(src.CourseStatus, true));
			// Configure Course to CourseResponseDto mapping with enum to string conversion
			config.NewConfig<Course, CourseResponseDto>()
				.Map(dest => dest.semster, src => src.semster.ToString())
				.Map(dest => dest.CourseStatus, src => src.CourseStatus.ToString());

			// Configure AssesmentDto to Assessment mapping with enum conversion
			config.NewConfig<AssesmentDto, Assessment>()
				.Map(dest => dest.AssType, src => Enum.Parse<AssTypes>(src.AssType, true));

			// Configure Assessment to AssesmentDto mapping with enum to string conversion
			config.NewConfig<Assessment, AssesmentDto>()
				.Map(dest => dest.AssType, src => src.AssType.ToString());
			// Mapping Email To UserName in ApplicationUser
			config.NewConfig<RegisterRequest, ApplicationUser>()
			.Map(dest => dest.UserName, src => src.Email);

			config.NewConfig<(ApplicationUser user, IList<string> roles), UserResponse>()
			.Map(dest => dest, src => src.user)
			.Map(dest => dest.Roles, src => src.roles)
			.Map(dest => dest.AcademicYear, src => src.user.AcademicYear.HasValue ? src.user.AcademicYear.Value.ToString() : null)
			.Map(dest => dest.DepartmentId , src => src.user.DepartmentId ?? 0);

			config.NewConfig<CreateUserRequest, ApplicationUser>()
			.Map(dest => dest.UserName, src => src.Email)
			.Map(dest => dest.EmailConfirmed, src => true)
			;

			config.NewConfig<UpdateUserRequest, ApplicationUser>()
				.Map(dest => dest.UserName, src => src.Email)
				.Map(dest => dest.NormalizedUserName, src => src.Email.ToUpper());

		}
	}
}
