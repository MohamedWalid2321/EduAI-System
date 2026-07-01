global using System;
global using System.Collections.Generic;
global using System.Linq;
global using System.Text;
global using System.Threading.Tasks;

global using Microsoft.AspNetCore.Mvc;
global using Microsoft.AspNetCore.Authorization;
global using ServiceAbstractionLayer;
global using Shared.Dtos;
global using Microsoft.AspNetCore.Http;
global using PresentationLayer.Attributes;
global using Shared.Dtos.CourseDto.Request;
global using Shared.Dtos.ContentDto.ContentRequest;
global using Shared.Dtos.AuthDto.Request;
global using Shared.Dtos.AssigmentDto.Request;

global using Microsoft.AspNetCore.Mvc.Filters;
global using Microsoft.Extensions.DependencyInjection;

global using PresentationLayer.Extensions;
global using Shared.Dtos.UserDto.Request;
global using Shared.Constants;
global using Microsoft.Extensions.Options;
global using System.Security.Claims;
global using Shared.Dtos.RiskAnalysisDto.Request;
global using Shared.Dtos.RiskAnalysisDto.Response;
global using Shared.Dtos.ContactDto;