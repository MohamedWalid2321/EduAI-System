<<<<<<< HEAD
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
=======
﻿namespace ServiceAbstractionLayer
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
{
	public interface IServiceManager
	{
		IDepartmentService DepartmentService { get; }
		ICourseService CourseService { get; }
		IContentService ContentService { get; }
		IAssigmentService AssignmentService { get; }
<<<<<<< HEAD
=======
		IAuthunticationService AuthunticationService { get; }
		IUserService UserService { get; }
		IRoleService RoleService { get; }


>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
	}
}
